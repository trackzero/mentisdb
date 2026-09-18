//! Extractive consolidation: turns uncovered summary-candidate windows into
//! `Summary`/[`ThoughtRole::Dream`] digests.
//!
//! This reuses [`crate::MentisDb`]'s existing `summary_candidates_from_thoughts`
//! (built on [`crate::search::build_summary_candidates`]) for window
//! selection and coverage skipping, and [`super::salience`] to rank
//! candidates and pick which member statements make the digest.

use super::{prompts, salience, ChatCompleter, DreamConfig};
use crate::search::{SummaryBuildConfig, ThoughtLocator};
use crate::{
    LlmExtractionConfig, MentisDb, Thought, ThoughtInput, ThoughtRelation, ThoughtRelationKind,
    ThoughtRole, ThoughtType,
};
use chrono::Utc;
use serde::Deserialize;
use uuid::Uuid;

const DIGEST_TOP_MEMBERS: usize = 5;
const DIGEST_STATEMENT_CHAR_CAP: usize = 280;
const DIGEST_TOTAL_CHAR_CAP: usize = 2000;
/// Confidence for a purely extractive digest (bullet-joined member content,
/// no invention risk).
const EXTRACTIVE_CONFIDENCE: f32 = 0.7;
/// Confidence for an LLM-rewritten ("abstractive") digest, per the design's
/// explicit "LLM consolidation <= 0.6" — between extractive's 0.7 (higher
/// trust, no invention) and recombination's 0.4 (speculative connection).
const ABSTRACTIVE_CONFIDENCE: f32 = 0.6;
/// Maximum length of an LLM-generated summary before it's rejected in favor
/// of falling back to the extractive digest.
const ABSTRACTIVE_SUMMARY_CHAR_CAP: usize = 500;

#[derive(Debug, Deserialize)]
struct ConsolidationResponse {
    summary: String,
}

/// Output of [`plan_consolidations`]: the planned summary thoughts plus any
/// LLM tokens spent producing them (`0` when `llm` was `None`).
#[derive(Debug, Default)]
pub(crate) struct ConsolidationOutcome {
    pub(crate) inputs: Vec<ThoughtInput>,
    pub(crate) tokens_used: u64,
}

/// Plan (but do not append) extractive or abstractive consolidation
/// summaries for the scan window `[scan_start_index, scan_end_index)`, up to
/// `budget` windows.
///
/// Pure planning: nothing is appended here, so this is safe to call for both
/// `dry_run` and real passes — the caller decides whether to append the
/// returned inputs.
pub(crate) async fn plan_consolidations(
    db: &MentisDb,
    config: &DreamConfig,
    scan_start_index: u64,
    scan_end_index: u64,
    pass_id: Uuid,
    budget: usize,
    llm: Option<(&LlmExtractionConfig, &dyn ChatCompleter)>,
) -> ConsolidationOutcome {
    if budget == 0 || scan_start_index >= scan_end_index {
        return ConsolidationOutcome::default();
    }

    let window: Vec<&Thought> = db.thoughts()[scan_start_index as usize..scan_end_index as usize]
        .iter()
        .filter(|thought| thought.role != ThoughtRole::Dream && !db.is_invalidated(thought.id))
        .collect();
    if window.is_empty() {
        return ConsolidationOutcome::default();
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
        return ConsolidationOutcome::default();
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

    let mut outcome = ConsolidationOutcome::default();
    for (_, candidate) in ranked.into_iter().take(budget) {
        let (input, tokens) =
            build_summary_input(db, &member_salience, &candidate, pass_id, llm).await;
        outcome.tokens_used += tokens;
        if let Some(input) = input {
            outcome.inputs.push(input);
        }
    }
    outcome
}

async fn build_summary_input(
    db: &MentisDb,
    member_salience: &(dyn Fn(&Thought) -> f32 + Sync),
    candidate: &crate::search::SummaryCandidate,
    pass_id: Uuid,
    llm: Option<(&LlmExtractionConfig, &dyn ChatCompleter)>,
) -> (Option<ThoughtInput>, u64) {
    let members: Vec<&Thought> = candidate
        .source_indices
        .iter()
        .filter_map(|&index| db.thoughts().get(index as usize))
        .collect();
    if members.is_empty() {
        return (None, 0);
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
    let extractive_digest: String = full_digest.chars().take(DIGEST_TOTAL_CHAR_CAP).collect();

    // When an LLM is configured, try to upgrade the extractive bullets into a
    // coherent gist. Any failure (parse error, empty/over-length summary)
    // falls back to the extractive digest rather than failing consolidation
    // outright — an LLM hiccup shouldn't cost a consolidation opportunity.
    let mut tokens_used = 0u64;
    let (digest, confidence, used_llm) = match llm {
        Some((llm_config, completer)) => {
            let (summary, tokens) = try_abstractive_summary(&bullets, llm_config, completer).await;
            tokens_used += tokens;
            match summary {
                Some(summary) => (summary, ABSTRACTIVE_CONFIDENCE, true),
                None => (extractive_digest, EXTRACTIVE_CONFIDENCE, false),
            }
        }
        None => (extractive_digest, EXTRACTIVE_CONFIDENCE, false),
    };

    let mut tags: Vec<String> = members
        .iter()
        .flat_map(|t| t.tags.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    tags.push("dream".to_string());
    tags.push(format!("dream:pass:{pass_id}"));
    tags.push("dream:consolidation".to_string());
    if used_llm {
        tags.push("dream:consolidation:llm".to_string());
    }

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

    let input = ThoughtInput::new(ThoughtType::Summary, digest)
        .with_role(ThoughtRole::Dream)
        .with_tags(tags)
        .with_concepts(concepts)
        .with_importance(max_importance)
        .with_confidence(confidence)
        .with_relations(relations);
    (Some(input), tokens_used)
}

/// Ask the LLM to rewrite extractive bullet points into one coherent gist.
/// Returns `(None, tokens_used)` on any failure (API error, non-JSON
/// response, empty or over-length summary), signaling the caller to fall
/// back to the extractive digest; `tokens_used` is still reported even on
/// failure since the call itself was still made.
async fn try_abstractive_summary(
    bullets: &[String],
    llm_config: &LlmExtractionConfig,
    completer: &dyn ChatCompleter,
) -> (Option<String>, u64) {
    let prompt = prompts::build_consolidation_prompt(bullets);
    let (raw, usage) = match completer
        .complete(llm_config, prompts::CONSOLIDATION_SYSTEM, &prompt, 0.2)
        .await
    {
        Ok(pair) => pair,
        Err(_) => return (None, 0),
    };
    let tokens_used = usage.total_tokens as u64;
    let Ok(parsed) = serde_json::from_str::<ConsolidationResponse>(raw.trim()) else {
        return (None, tokens_used);
    };
    let summary = parsed.summary.trim();
    if summary.is_empty() || summary.chars().count() > ABSTRACTIVE_SUMMARY_CHAR_CAP {
        return (None, tokens_used);
    }
    (Some(summary.to_string()), tokens_used)
}
