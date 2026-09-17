//! Extractive consolidation: turns uncovered summary-candidate windows into
//! `Summary`/[`ThoughtRole::Dream`] digests.
//!
//! This reuses [`crate::MentisDb`]'s existing `summary_candidates_from_thoughts`
//! (built on [`crate::search::build_summary_candidates`]) for window
//! selection and coverage skipping, and [`super::salience`] to rank
//! candidates and pick which member statements make the digest.

use super::{salience, DreamConfig};
use crate::search::{SummaryBuildConfig, ThoughtLocator};
use crate::{
    MentisDb, Thought, ThoughtInput, ThoughtRelation, ThoughtRelationKind, ThoughtRole, ThoughtType,
};
use chrono::Utc;
use uuid::Uuid;

const DIGEST_TOP_MEMBERS: usize = 5;
const DIGEST_STATEMENT_CHAR_CAP: usize = 280;
const DIGEST_TOTAL_CHAR_CAP: usize = 2000;

/// Plan (but do not append) extractive consolidation summaries for the scan
/// window `[scan_start_index, scan_end_index)`, up to `budget` windows.
///
/// Pure planning: nothing is appended here, so this is safe to call for both
/// `dry_run` and real passes — the caller decides whether to append the
/// returned inputs.
pub(crate) fn plan_consolidations(
    db: &MentisDb,
    config: &DreamConfig,
    scan_start_index: u64,
    scan_end_index: u64,
    pass_id: Uuid,
    budget: usize,
) -> Vec<ThoughtInput> {
    if budget == 0 || scan_start_index >= scan_end_index {
        return Vec::new();
    }

    let window: Vec<&Thought> = db.thoughts()[scan_start_index as usize..scan_end_index as usize]
        .iter()
        .filter(|thought| thought.role != ThoughtRole::Dream && !db.is_invalidated(thought.id))
        .collect();
    if window.is_empty() {
        return Vec::new();
    }

    // A window of exactly one member has nothing to consolidate: the
    // "summary" would just restate the single source thought while still
    // spending a Summarizes relation and a write-budget slot. Requiring at
    // least two members keeps consolidation meaningful.
    let candidates: Vec<_> = db
        .summary_candidates_from_thoughts(&window, SummaryBuildConfig::default())
        .into_iter()
        .filter(|candidate| candidate.source_indices.len() >= 2)
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }

    let adjacency = db.cached_adjacency_index();
    let now = Utc::now();

    let member_salience = |thought: &Thought| -> f32 {
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

    let mut ranked: Vec<(f32, crate::search::SummaryCandidate)> = candidates
        .into_iter()
        .map(|candidate| {
            let summed: f32 = candidate
                .source_indices
                .iter()
                .filter_map(|&index| db.thoughts().get(index as usize))
                .map(member_salience)
                .sum();
            (summed, candidate)
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.start_index.cmp(&right.1.start_index))
    });

    ranked
        .into_iter()
        .take(budget)
        .filter_map(|(_, candidate)| build_summary_input(db, &member_salience, &candidate, pass_id))
        .collect()
}

fn build_summary_input(
    db: &MentisDb,
    member_salience: &dyn Fn(&Thought) -> f32,
    candidate: &crate::search::SummaryCandidate,
    pass_id: Uuid,
) -> Option<ThoughtInput> {
    let members: Vec<&Thought> = candidate
        .source_indices
        .iter()
        .filter_map(|&index| db.thoughts().get(index as usize))
        .collect();
    if members.is_empty() {
        return None;
    }

    let mut by_salience: Vec<(&Thought, f32)> =
        members.iter().map(|&t| (t, member_salience(t))).collect();
    by_salience.sort_by(|left, right| right.1.total_cmp(&left.1));

    let mut seen_statements: Vec<String> = Vec::new();
    let mut bullets: Vec<String> = Vec::new();
    for (thought, _) in by_salience.iter().take(DIGEST_TOP_MEMBERS) {
        let normalized = thought.content.trim().to_lowercase();
        if seen_statements.contains(&normalized) {
            continue;
        }
        seen_statements.push(normalized);
        let truncated: String = thought
            .content
            .chars()
            .take(DIGEST_STATEMENT_CHAR_CAP)
            .collect();
        bullets.push(format!("- {truncated}"));
    }

    let header = format!(
        "Extractive summary of {} thoughts (indices {}..={})",
        members.len(),
        candidate.start_index,
        candidate.end_index
    );
    let full_digest = format!("{header}\n{}", bullets.join("\n"));
    let digest: String = full_digest.chars().take(DIGEST_TOTAL_CHAR_CAP).collect();

    let mut tags: Vec<String> = members
        .iter()
        .flat_map(|t| t.tags.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    tags.push("dream".to_string());
    tags.push(format!("dream:pass:{pass_id}"));
    tags.push("dream:consolidation".to_string());

    let concepts: Vec<String> = members
        .iter()
        .flat_map(|t| t.concepts.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let max_importance = members.iter().map(|t| t.importance).fold(0.0f32, f32::max);

    let relations: Vec<ThoughtRelation> = members
        .iter()
        .map(|t| ThoughtRelation::new(ThoughtRelationKind::Summarizes, t.id))
        .collect();

    Some(
        ThoughtInput::new(ThoughtType::Summary, digest)
            .with_role(ThoughtRole::Dream)
            .with_tags(tags)
            .with_concepts(concepts)
            .with_importance(max_importance)
            .with_confidence(0.7)
            .with_relations(relations),
    )
}
