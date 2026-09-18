//! Dedup suggestions: proposes that one of a pair of near-duplicate thoughts
//! from the same agent likely supersedes the other, without ever appending
//! `Supersedes`/`Invalidates`/`Corrects` itself.
//!
//! A pair qualifies only when it clears ALL of: high lexical overlap, high
//! vector cosine similarity, written by the same agent, and not already
//! linked by any relation in either direction. Suggestions are always plain
//! `Finding`/[`ThoughtRole::Dream`] thoughts carrying two `RelatedTo`
//! relations — an awake agent or human decides whether to actually promote
//! one over the other.

use super::DreamConfig;
use crate::{
    MentisDb, Thought, ThoughtInput, ThoughtRelation, ThoughtRelationKind, ThoughtRole, ThoughtType,
};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Minimum Jaccard lexical overlap (on normalized/stemmed tokens) for a pair
/// to be considered "high lexical overlap." Not specified numerically by the
/// design doc; `0.5` is a majority-overlap bar independent of the unrelated
/// `dedup_threshold`/`auto_edge_threshold` config knobs.
const LEXICAL_OVERLAP_THRESHOLD: f32 = 0.5;
/// Minimum vector cosine similarity for a pair to be considered a near
/// duplicate, per the design doc's own fixed number (no config override).
///
/// `pub(crate)` so `dream::recombine`'s contradiction-check can define its
/// similarity band relative to this exact value (contradictions are
/// "high-similarity pairs phase 1 did not flag as duplicates," i.e. below
/// this threshold).
pub(crate) const COSINE_THRESHOLD: f32 = 0.95;
/// Confidence assigned to every dedup suggestion: a suggestion about two
/// thoughts, not a digest of them, so deliberately below consolidation's 0.7
/// extractive cap.
const SUGGESTION_CONFIDENCE: f32 = 0.6;

/// Jaccard similarity between the normalized, stemmed token sets of two
/// pieces of content. Parallels (does not call into or modify) the existing
/// inline dedup-on-append Jaccard logic in [`crate::MentisDb::append_thought`].
pub(crate) fn jaccard_overlap(a: &str, b: &str) -> f32 {
    let tokens_a: HashSet<String> = crate::search::lexical::normalize_lexical_tokens(a, true)
        .into_iter()
        .collect();
    let tokens_b: HashSet<String> = crate::search::lexical::normalize_lexical_tokens(b, true)
        .into_iter()
        .collect();
    if tokens_a.is_empty() && tokens_b.is_empty() {
        return 0.0;
    }
    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Plan (but do not append) dedup suggestions for the scan window
/// `[scan_start_index, scan_end_index)`, up to `budget` suggestions.
///
/// Pure planning: nothing is appended here, so this is safe to call for both
/// `dry_run` and real passes.
///
/// If the chain has no managed vector sidecar at all, dedup-suggestion
/// generation is skipped entirely for this pass (an empty result, not an
/// error) — cosine similarity cannot be computed without embeddings, and the
/// no-embedding core must still work.
pub(crate) fn plan_dedup_suggestions(
    db: &MentisDb,
    _config: &DreamConfig,
    scan_start_index: u64,
    scan_end_index: u64,
    pass_id: Uuid,
    budget: usize,
) -> Vec<ThoughtInput> {
    if budget == 0 || scan_start_index >= scan_end_index {
        return Vec::new();
    }

    let metadata = match db.managed_vector_sidecars().into_iter().next() {
        Some(metadata) => metadata,
        None => return Vec::new(),
    };
    let sidecar = match db.load_vector_sidecar(&metadata) {
        Ok(Some(sidecar)) => sidecar,
        _ => return Vec::new(),
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

    let mut by_agent: HashMap<&str, Vec<&Thought>> = HashMap::new();
    for thought in &window {
        by_agent
            .entry(thought.agent_id.as_str())
            .or_default()
            .push(thought);
    }

    let mut suggestions = Vec::new();
    'agents: for group in by_agent.values() {
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let (a, b) = (group[i], group[j]);
                if suggestions.len() >= budget {
                    break 'agents;
                }
                if already_linked(a, b) {
                    continue;
                }
                let lexical = jaccard_overlap(&a.content, &b.content);
                if lexical < LEXICAL_OVERLAP_THRESHOLD {
                    continue;
                }
                let (Some(vec_a), Some(vec_b)) = (vectors.get(&a.id), vectors.get(&b.id)) else {
                    continue;
                };
                let Some(cosine) = crate::search::vector::cosine_similarity(vec_a, vec_b) else {
                    continue;
                };
                if cosine < COSINE_THRESHOLD {
                    continue;
                }
                suggestions.push(build_suggestion_input(a, b, lexical, cosine, pass_id));
            }
        }
    }
    suggestions
}

fn already_linked(a: &Thought, b: &Thought) -> bool {
    a.relations.iter().any(|r| r.target_id == b.id)
        || b.relations.iter().any(|r| r.target_id == a.id)
}

fn build_suggestion_input(
    a: &Thought,
    b: &Thought,
    lexical: f32,
    cosine: f32,
    pass_id: Uuid,
) -> ThoughtInput {
    let (winner, loser) = if a.importance > b.importance {
        (a, b)
    } else if b.importance > a.importance || b.timestamp > a.timestamp {
        (b, a)
    } else {
        (a, b)
    };

    let content = format!(
        "{} and {} look like near-duplicates (cosine={cosine:.3}, lexical={lexical:.3}); {} \
         (higher importance/newer) likely supersedes {}. Suggestion only \u{2014} nothing has \
         been superseded automatically.",
        a.id, b.id, winner.id, loser.id
    );

    ThoughtInput::new(ThoughtType::Finding, content)
        .with_role(ThoughtRole::Dream)
        .with_tags(vec![
            "dream".to_string(),
            format!("dream:pass:{pass_id}"),
            "dream:suggestion".to_string(),
        ])
        .with_importance(f32::max(a.importance, b.importance))
        .with_confidence(SUGGESTION_CONFIDENCE)
        .with_relations(vec![
            ThoughtRelation::new(ThoughtRelationKind::RelatedTo, a.id),
            ThoughtRelation::new(ThoughtRelationKind::RelatedTo, b.id),
        ])
}
