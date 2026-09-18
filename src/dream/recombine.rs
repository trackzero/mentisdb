//! Phase 2 LLM-assisted recombination and contradiction-checking.
//!
//! Recombination is the REM analog: cluster the scan window's high-salience
//! thoughts by shared tag/concept, find pairs of clusters whose vector
//! centroids are "distant" (low cosine), and ask the LLM to propose at most a
//! few non-obvious, testable connections between them. Contradiction-check
//! looks at pairs of thoughts that are similar-but-not-duplicate (a band just
//! below Phase 1 dedup's cosine threshold) and asks the LLM whether they
//! directly contradict each other.
//!
//! Both operations are opt-in (only run when `DreamConfig.llm` is set),
//! share one LLM-call budget (`DreamConfig.recombination_budget`), and never
//! append `Contradicts`/`Supersedes`/`Invalidates` themselves — only
//! `RelatedTo`/`DerivedFrom` suggestions, exactly like Phase 1's dedup
//! suggestions.

use super::prompts;
use super::suggest::COSINE_THRESHOLD as DEDUP_COSINE_THRESHOLD;
use super::{salience, ChatCompleter, DreamConfig};
use crate::search::ThoughtLocator;
use crate::{
    LlmExtractionConfig, MentisDb, Thought, ThoughtInput, ThoughtRelation, ThoughtRelationKind,
    ThoughtRole, ThoughtType,
};
use chrono::Utc;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Cluster centroid pairs at or below this cosine similarity are "distant"
/// enough for recombination. Chosen to sit clearly below both the dedup
/// threshold (0.95) and the contradiction band (0.75-0.95), so no pair can be
/// flagged by more than one Phase 1/2 mechanism.
const CLUSTER_DISTANCE_MAX_COSINE: f32 = 0.3;
/// Lower bound of the contradiction-check similarity band: below this, pairs
/// are typically just topically related, not comparable enough for a
/// meaningful contradiction judgment.
const CONTRADICTION_MIN_COSINE: f32 = 0.75;
/// Upper bound of the contradiction-check band: matches Phase 1 dedup's own
/// threshold exactly, since cosine >= 0.95 is already dedup's territory.
const CONTRADICTION_MAX_COSINE: f32 = DEDUP_COSINE_THRESHOLD;
/// Cosine similarity at or above which a recombination candidate is
/// considered a near-duplicate of existing content and rejected, per the
/// design's literal novelty-filter threshold.
const NOVELTY_MAX_COSINE: f32 = 0.9;
/// Maximum content length for any Phase 2 LLM-generated thought (recombination
/// content, contradiction rationale). Not specified numerically by the
/// design; bounds storage/prompt-cost blowback while leaving room for a real
/// claim.
const CONTENT_CHAR_CAP: usize = 500;
/// Confidence assigned to recombination outputs, per the design's explicit
/// "confidence <= 0.4" cap for recombination.
const RECOMBINATION_CONFIDENCE: f32 = 0.4;
/// Confidence assigned to contradiction suggestions, per the design.
const CONTRADICTION_CONFIDENCE: f32 = 0.4;
/// Members per cluster side shown to the model in one recombination call.
/// Bounds prompt size; not specified numerically by the design.
const CLUSTER_TOP_MEMBERS: usize = 5;
/// Maximum connections the model may propose in one recombination call.
const MAX_CONNECTIONS_PER_CALL: usize = 3;

/// Allowed `ThoughtType`s for a recombination output, per the design.
const RECOMBINATION_ALLOWED_TYPES: [ThoughtType; 4] = [
    ThoughtType::Hypothesis,
    ThoughtType::Idea,
    ThoughtType::Wonder,
    ThoughtType::Question,
];

/// Combined result of one pass's recombination + contradiction-check work.
#[derive(Debug, Default)]
pub(crate) struct Phase2ExploratoryOutcome {
    pub(crate) recombinations: Vec<ThoughtInput>,
    pub(crate) contradictions: Vec<ThoughtInput>,
    pub(crate) rejections: u64,
    pub(crate) tokens_used: u64,
}

#[derive(Debug, Deserialize)]
struct RawConnection {
    content: String,
    thought_type: String,
    #[serde(default)]
    cites: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct RecombinationResponse {
    connections: Vec<RawConnection>,
}

#[derive(Debug, Deserialize)]
struct ContradictionResponse {
    contradicts: bool,
    #[serde(default)]
    rationale: String,
}

/// Plan (but do not append) recombination and contradiction-check outputs for
/// the scan window `[scan_start_index, scan_end_index)`.
///
/// `write_budget` and `call_budget` are two independent caps: `write_budget`
/// comes from the pass's overall `max_writes_per_pass` (shared with
/// consolidation/dedup, decremented only on actual writes); `call_budget`
/// comes from `DreamConfig.recombination_budget` (decremented on every LLM
/// call *attempt*, including ones that end in rejection, so a broken LLM
/// can't stall a pass in an unbounded retry loop). Both recombination and
/// contradiction-check draw from the same `call_budget` pool; recombination
/// runs first since its candidates are already ranked by salience.
///
/// If the chain has no managed vector sidecar at all, both operations are
/// skipped entirely (an empty outcome, not an error) — neither clustering
/// distance nor contradiction similarity can be computed without embeddings.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn plan_recombination_and_contradictions(
    db: &MentisDb,
    config: &DreamConfig,
    llm_config: &LlmExtractionConfig,
    completer: &dyn ChatCompleter,
    scan_start_index: u64,
    scan_end_index: u64,
    pass_id: Uuid,
    write_budget: usize,
    call_budget: usize,
) -> Phase2ExploratoryOutcome {
    let mut outcome = Phase2ExploratoryOutcome::default();
    if write_budget == 0 || call_budget == 0 || scan_start_index >= scan_end_index {
        return outcome;
    }

    let metadata = match db.managed_vector_sidecars().into_iter().next() {
        Some(metadata) => metadata,
        None => return outcome,
    };
    let sidecar = match db.load_vector_sidecar(&metadata) {
        Ok(Some(sidecar)) => sidecar,
        _ => return outcome,
    };
    let vectors: HashMap<Uuid, &[f32]> = sidecar
        .entries
        .iter()
        .map(|entry| (entry.thought_id, entry.vector.as_slice()))
        .collect();

    let window: Vec<&Thought> = db.thoughts()[scan_start_index as usize..scan_end_index as usize]
        .iter()
        .filter(|thought| thought.role != ThoughtRole::Dream && !db.is_invalidated(thought.id))
        .collect();
    if window.len() < 2 {
        return outcome;
    }

    let adjacency = db.cached_adjacency_index();
    let now = Utc::now();
    let salience_of = |thought: &Thought| -> f32 {
        let locator = ThoughtLocator::local(thought);
        let effective_ts =
            salience::effective_timestamp(thought, &locator, &adjacency, db.thoughts());
        let age_days = (now - effective_ts).num_seconds().max(0) as f32 / 86400.0;
        let half_life = salience::resolve_half_life_days(
            thought.role,
            thought.thought_type,
            &config.half_life_overrides_days,
        );
        let in_degree = salience::in_degree(&locator, &adjacency);
        salience::salience(
            thought.importance,
            thought.thought_type,
            age_days,
            half_life,
            in_degree,
        )
    };

    let mut write_budget = write_budget;
    let mut call_budget = call_budget;

    run_recombination(
        db,
        &metadata,
        &window,
        &vectors,
        &salience_of,
        llm_config,
        completer,
        pass_id,
        &mut write_budget,
        &mut call_budget,
        &mut outcome,
    )
    .await;

    run_contradiction_check(
        db,
        &window,
        &vectors,
        llm_config,
        completer,
        pass_id,
        &mut write_budget,
        &mut call_budget,
        &mut outcome,
    )
    .await;

    outcome
}

#[allow(clippy::too_many_arguments)]
async fn run_recombination<'a>(
    db: &MentisDb,
    metadata: &crate::search::EmbeddingMetadata,
    window: &[&'a Thought],
    vectors: &HashMap<Uuid, &[f32]>,
    salience_of: &(dyn Fn(&Thought) -> f32 + Sync),
    llm_config: &LlmExtractionConfig,
    completer: &dyn ChatCompleter,
    pass_id: Uuid,
    write_budget: &mut usize,
    call_budget: &mut usize,
    outcome: &mut Phase2ExploratoryOutcome,
) {
    let saliences: Vec<f32> = window.iter().map(|t| salience_of(t)).collect();
    let median = median_of(&saliences);

    // Group thoughts scoring above the median salience by every tag/concept
    // they carry. A thought may land in multiple clusters.
    let mut clusters: HashMap<String, Vec<&Thought>> = HashMap::new();
    for (thought, score) in window.iter().zip(saliences.iter()) {
        if *score <= median {
            continue;
        }
        for key in thought.tags.iter().chain(thought.concepts.iter()) {
            clusters.entry(key.clone()).or_default().push(thought);
        }
    }
    clusters.retain(|_, members| members.len() >= 2);
    if clusters.len() < 2 {
        return;
    }

    let centroids: Vec<(String, Vec<&Thought>, Vec<f32>)> = clusters
        .into_iter()
        .filter_map(|(key, members)| {
            let centroid = centroid_of(&members, vectors)?;
            Some((key, members, centroid))
        })
        .collect();
    if centroids.len() < 2 {
        return;
    }

    let mut candidate_pairs: Vec<(f32, usize, usize)> = Vec::new();
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let Some(cosine) =
                crate::search::vector::cosine_similarity(&centroids[i].2, &centroids[j].2)
            else {
                continue;
            };
            if cosine > CLUSTER_DISTANCE_MAX_COSINE {
                continue;
            }
            let combined_salience: f32 = centroids[i]
                .1
                .iter()
                .chain(centroids[j].1.iter())
                .map(|t| salience_of(t))
                .sum();
            candidate_pairs.push((combined_salience, i, j));
        }
    }
    candidate_pairs.sort_by(|left, right| right.0.total_cmp(&left.0));

    for (_, i, j) in candidate_pairs {
        if *write_budget == 0 || *call_budget == 0 {
            break;
        }

        let mut label_map: Vec<Uuid> = Vec::new();
        let cluster_a = top_members_labeled(&centroids[i].1, salience_of, &mut label_map);
        let cluster_b = top_members_labeled(&centroids[j].1, salience_of, &mut label_map);
        let cluster_a_refs: Vec<(usize, &str)> = cluster_a
            .iter()
            .map(|(label, content)| (*label, content.as_str()))
            .collect();
        let cluster_b_refs: Vec<(usize, &str)> = cluster_b
            .iter()
            .map(|(label, content)| (*label, content.as_str()))
            .collect();
        let prompt = prompts::build_recombination_prompt(
            &cluster_a_refs,
            &cluster_b_refs,
            MAX_CONNECTIONS_PER_CALL,
        );

        *call_budget -= 1;
        let response = completer
            .complete(llm_config, prompts::RECOMBINATION_SYSTEM, &prompt, 0.3)
            .await;
        let (raw, usage) = match response {
            Ok(pair) => pair,
            Err(_) => {
                outcome.rejections += 1;
                continue;
            }
        };
        outcome.tokens_used += usage.total_tokens as u64;

        let parsed: RecombinationResponse = match serde_json::from_str(raw.trim()) {
            Ok(parsed) => parsed,
            Err(_) => {
                outcome.rejections += 1;
                continue;
            }
        };

        for connection in parsed
            .connections
            .into_iter()
            .take(MAX_CONNECTIONS_PER_CALL)
        {
            if *write_budget == 0 {
                break;
            }
            match validate_connection(&connection, &label_map) {
                Some((thought_type, cited_ids)) => {
                    let avg_importance = average_importance(window, &cited_ids);
                    let candidate = ThoughtInput::new(thought_type, connection.content.clone())
                        .with_role(ThoughtRole::Dream)
                        .with_tags(vec![
                            "dream".to_string(),
                            format!("dream:pass:{pass_id}"),
                            "dream:recombination".to_string(),
                        ])
                        .with_importance(avg_importance)
                        .with_confidence(RECOMBINATION_CONFIDENCE)
                        .with_relations(
                            cited_ids
                                .iter()
                                .map(|id| {
                                    ThoughtRelation::new(ThoughtRelationKind::DerivedFrom, *id)
                                })
                                .collect(),
                        );

                    if is_near_existing(db, metadata, window, vectors, &connection.content) {
                        outcome.rejections += 1;
                        continue;
                    }

                    *write_budget -= 1;
                    outcome.recombinations.push(candidate);
                }
                None => outcome.rejections += 1,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_contradiction_check(
    db: &MentisDb,
    window: &[&Thought],
    vectors: &HashMap<Uuid, &[f32]>,
    llm_config: &LlmExtractionConfig,
    completer: &dyn ChatCompleter,
    pass_id: Uuid,
    write_budget: &mut usize,
    call_budget: &mut usize,
    outcome: &mut Phase2ExploratoryOutcome,
) {
    let already_flagged = existing_contradiction_pairs(db);

    for i in 0..window.len() {
        for j in (i + 1)..window.len() {
            if *write_budget == 0 || *call_budget == 0 {
                return;
            }
            let (a, b) = (window[i], window[j]);
            let (Some(vec_a), Some(vec_b)) = (vectors.get(&a.id), vectors.get(&b.id)) else {
                continue;
            };
            let Some(cosine) = crate::search::vector::cosine_similarity(vec_a, vec_b) else {
                continue;
            };
            if !(CONTRADICTION_MIN_COSINE..CONTRADICTION_MAX_COSINE).contains(&cosine) {
                continue;
            }
            let key = canonical_pair(a.id, b.id);
            if already_flagged.contains(&key) {
                continue;
            }

            let prompt = prompts::build_contradiction_prompt(1, &a.content, 2, &b.content);
            *call_budget -= 1;
            let response = completer
                .complete(llm_config, prompts::CONTRADICTION_SYSTEM, &prompt, 0.1)
                .await;
            let (raw, usage) = match response {
                Ok(pair) => pair,
                Err(_) => {
                    outcome.rejections += 1;
                    continue;
                }
            };
            outcome.tokens_used += usage.total_tokens as u64;

            let parsed: ContradictionResponse = match serde_json::from_str(raw.trim()) {
                Ok(parsed) => parsed,
                Err(_) => {
                    outcome.rejections += 1;
                    continue;
                }
            };
            if !parsed.contradicts {
                continue;
            }
            let rationale = parsed.rationale.trim();
            if rationale.is_empty() || rationale.chars().count() > CONTENT_CHAR_CAP {
                outcome.rejections += 1;
                continue;
            }

            *write_budget -= 1;
            outcome.contradictions.push(
                ThoughtInput::new(ThoughtType::Surprise, rationale.to_string())
                    .with_role(ThoughtRole::Dream)
                    .with_tags(vec![
                        "dream".to_string(),
                        format!("dream:pass:{pass_id}"),
                        "dream:contradiction".to_string(),
                    ])
                    .with_confidence(CONTRADICTION_CONFIDENCE)
                    .with_relations(vec![
                        ThoughtRelation::new(ThoughtRelationKind::RelatedTo, a.id),
                        ThoughtRelation::new(ThoughtRelationKind::RelatedTo, b.id),
                    ]),
            );
        }
    }
}

fn median_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn centroid_of(members: &[&Thought], vectors: &HashMap<Uuid, &[f32]>) -> Option<Vec<f32>> {
    let member_vectors: Vec<&[f32]> = members
        .iter()
        .filter_map(|t| vectors.get(&t.id).copied())
        .collect();
    if member_vectors.is_empty() {
        return None;
    }
    let dimension = member_vectors[0].len();
    let mut sum = vec![0.0f32; dimension];
    let mut count = 0usize;
    for vector in &member_vectors {
        if vector.len() != dimension {
            continue;
        }
        for (accumulator, value) in sum.iter_mut().zip(vector.iter()) {
            *accumulator += value;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    for value in &mut sum {
        *value /= count as f32;
    }
    Some(sum)
}

fn top_members_labeled(
    members: &[&Thought],
    salience_of: &(dyn Fn(&Thought) -> f32 + Sync),
    label_map: &mut Vec<Uuid>,
) -> Vec<(usize, String)> {
    let mut ranked: Vec<&&Thought> = members.iter().collect();
    ranked.sort_by(|a, b| salience_of(b).total_cmp(&salience_of(a)));
    ranked
        .into_iter()
        .take(CLUSTER_TOP_MEMBERS)
        .map(|thought| {
            label_map.push(thought.id);
            (label_map.len(), thought.content.clone())
        })
        .collect()
}

fn validate_connection(
    connection: &RawConnection,
    label_map: &[Uuid],
) -> Option<(ThoughtType, Vec<Uuid>)> {
    let thought_type: ThoughtType = connection.thought_type.parse().ok()?;
    if !RECOMBINATION_ALLOWED_TYPES.contains(&thought_type) {
        return None;
    }
    let content = connection.content.trim();
    if content.is_empty() || content.chars().count() > CONTENT_CHAR_CAP {
        return None;
    }
    if connection.cites.is_empty() {
        return None;
    }
    let mut cited_ids = Vec::with_capacity(connection.cites.len());
    for &label in &connection.cites {
        if label == 0 || label > label_map.len() {
            return None;
        }
        cited_ids.push(label_map[label - 1]);
    }
    Some((thought_type, cited_ids))
}

fn average_importance(window: &[&Thought], cited_ids: &[Uuid]) -> f32 {
    let cited: Vec<f32> = window
        .iter()
        .filter(|t| cited_ids.contains(&t.id))
        .map(|t| t.importance)
        .collect();
    if cited.is_empty() {
        0.5
    } else {
        cited.iter().sum::<f32>() / cited.len() as f32
    }
}

/// Reject a recombination candidate whose generated content is a near-cosine
/// duplicate of an existing window thought, per the design's novelty filter.
/// Scoped to the scan window's own sidecar entries (not the whole chain) to
/// bound cost. Embeds `content` on the fly via
/// [`MentisDb::embed_text_with_managed_provider`], reusing the exact provider
/// that produced the window's existing vectors so the comparison is in the
/// same embedding space. If embedding fails for any reason, this fails
/// closed (treats the candidate as "too similar to tell" and rejects it),
/// consistent with the "LLM output is untrusted" posture.
fn is_near_existing(
    db: &MentisDb,
    metadata: &crate::search::EmbeddingMetadata,
    window: &[&Thought],
    vectors: &HashMap<Uuid, &[f32]>,
    content: &str,
) -> bool {
    let candidate_vector = match db.embed_text_with_managed_provider(metadata, content) {
        Ok(Some(vector)) => vector,
        _ => return true,
    };
    window.iter().any(|thought| {
        vectors
            .get(&thought.id)
            .and_then(|existing| {
                crate::search::vector::cosine_similarity(&candidate_vector, existing)
            })
            .is_some_and(|cosine| cosine >= NOVELTY_MAX_COSINE)
    })
}

fn canonical_pair(a: Uuid, b: Uuid) -> (Uuid, Uuid) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

fn existing_contradiction_pairs(db: &MentisDb) -> HashSet<(Uuid, Uuid)> {
    let mut pairs = HashSet::new();
    for thought in db.thoughts() {
        if thought.role != ThoughtRole::Dream
            || thought.thought_type != ThoughtType::Surprise
            || !thought.tags.iter().any(|t| t == "dream:contradiction")
        {
            continue;
        }
        let related: Vec<Uuid> = thought
            .relations
            .iter()
            .filter(|r| r.kind == ThoughtRelationKind::RelatedTo)
            .map(|r| r.target_id)
            .collect();
        if related.len() == 2 {
            pairs.insert(canonical_pair(related[0], related[1]));
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_computes_even_and_odd_length_correctly() {
        assert_eq!(median_of(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_of(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_of(&[]), 0.0);
    }

    #[test]
    fn canonical_pair_is_order_independent() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        assert_eq!(canonical_pair(a, b), canonical_pair(b, a));
    }

    #[test]
    fn validate_connection_rejects_unknown_type() {
        let label_map = vec![Uuid::new_v4()];
        let connection = RawConnection {
            content: "a claim".to_string(),
            thought_type: "Memory".to_string(),
            cites: vec![1],
        };
        assert!(validate_connection(&connection, &label_map).is_none());
    }

    #[test]
    fn validate_connection_rejects_out_of_range_citation() {
        let label_map = vec![Uuid::new_v4()];
        let connection = RawConnection {
            content: "a claim".to_string(),
            thought_type: "Hypothesis".to_string(),
            cites: vec![2],
        };
        assert!(validate_connection(&connection, &label_map).is_none());
    }

    #[test]
    fn validate_connection_rejects_over_length_content() {
        let label_map = vec![Uuid::new_v4()];
        let connection = RawConnection {
            content: "x".repeat(CONTENT_CHAR_CAP + 1),
            thought_type: "Idea".to_string(),
            cites: vec![1],
        };
        assert!(validate_connection(&connection, &label_map).is_none());
    }

    #[test]
    fn validate_connection_accepts_a_well_formed_connection() {
        let id = Uuid::new_v4();
        let label_map = vec![id];
        let connection = RawConnection {
            content: "a testable connection".to_string(),
            thought_type: "Wonder".to_string(),
            cites: vec![1],
        };
        let (thought_type, cited) = validate_connection(&connection, &label_map).unwrap();
        assert_eq!(thought_type, ThoughtType::Wonder);
        assert_eq!(cited, vec![id]);
    }

    #[test]
    fn centroid_of_averages_member_vectors_elementwise() {
        let dir = tempfile::tempdir().unwrap();
        let mut chain = MentisDb::open_with_key_and_storage_kind(
            dir.path(),
            "recombine-centroid-test",
            crate::StorageAdapterKind::Binary,
        )
        .unwrap();
        let a = chain
            .append_thought("agent", ThoughtInput::new(ThoughtType::Finding, "a"))
            .unwrap()
            .clone();
        let b = chain
            .append_thought("agent", ThoughtInput::new(ThoughtType::Finding, "b"))
            .unwrap()
            .clone();

        let mut vectors: HashMap<Uuid, &[f32]> = HashMap::new();
        let va: &[f32] = &[1.0, 3.0];
        let vb: &[f32] = &[3.0, 1.0];
        vectors.insert(a.id, va);
        vectors.insert(b.id, vb);

        let members = vec![&a, &b];
        let centroid = centroid_of(&members, &vectors).unwrap();
        assert_eq!(centroid, vec![2.0, 2.0]);
    }

    #[test]
    fn average_importance_averages_only_cited_members() {
        let dir = tempfile::tempdir().unwrap();
        let mut chain = MentisDb::open_with_key_and_storage_kind(
            dir.path(),
            "recombine-avg-importance-test",
            crate::StorageAdapterKind::Binary,
        )
        .unwrap();
        let a = chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, "a").with_importance(0.2),
            )
            .unwrap()
            .clone();
        let b = chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, "b").with_importance(0.8),
            )
            .unwrap()
            .clone();
        let c = chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, "c").with_importance(0.0),
            )
            .unwrap()
            .clone();

        let window = vec![&a, &b, &c];
        let avg = average_importance(&window, &[a.id, b.id]);
        assert!((avg - 0.5).abs() < 1e-6);
    }
}
