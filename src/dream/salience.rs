//! Salience scoring and query-time decay for offline dream passes.
//!
//! These are pure functions operating on plain data (no `&MentisDb`), so
//! extractive consolidation, dedup suggestions, and `use_decay` ranked-search
//! scoring can all share one implementation without repeatedly rebuilding a
//! chain's adjacency index.

use crate::search::{ThoughtAdjacencyIndex, ThoughtLocator};
use crate::{Thought, ThoughtRole, ThoughtType};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Half-life, in days, for types the design tiers as long-lived (~1 year).
pub const HALF_LIFE_LONG_DAYS: u32 = 365;
/// Half-life, in days, for types the design tiers as medium-lived (~1 quarter).
///
/// Also the default for any [`ThoughtType`] the design doc doesn't explicitly
/// tier.
pub const HALF_LIFE_MEDIUM_DAYS: u32 = 90;
/// Half-life, in days, for types/roles the design tiers as short-lived (~1 week).
pub const HALF_LIFE_SHORT_DAYS: u32 = 7;
/// Floor applied to [`effective_importance`] so query-time decay never drives
/// a thought's effective importance to exactly zero.
pub const DECAY_FLOOR: f32 = 0.05;

/// Look up the base query-time-decay half-life for a role/type combination.
///
/// Precedence (checked in this order — role first, then type):
/// 1. [`ThoughtRole::WorkingMemory`] is always [`HALF_LIFE_SHORT_DAYS`],
///    regardless of `thought_type`.
/// 2. Long-tier types (`Constraint`, `Decision`, `UserTrait`,
///    `PreferenceUpdate`, `LessonLearned`) are [`HALF_LIFE_LONG_DAYS`].
/// 3. Short-tier types (`Checkpoint`, `StateSnapshot`) are
///    [`HALF_LIFE_SHORT_DAYS`].
/// 4. Everything else, including `Finding`/`FactLearned`/`Insight` and every
///    type the design doc leaves untiered, is [`HALF_LIFE_MEDIUM_DAYS`].
///
/// Use [`resolve_half_life_days`] to additionally honor a per-type override.
///
/// # Example
///
/// ```
/// use mentisdb::dream::salience::{half_life_days, HALF_LIFE_LONG_DAYS};
/// use mentisdb::{ThoughtRole, ThoughtType};
///
/// assert_eq!(half_life_days(ThoughtRole::Memory, ThoughtType::Decision), HALF_LIFE_LONG_DAYS);
/// ```
pub fn half_life_days(role: ThoughtRole, thought_type: ThoughtType) -> u32 {
    if role == ThoughtRole::WorkingMemory {
        return HALF_LIFE_SHORT_DAYS;
    }
    match thought_type {
        ThoughtType::Constraint
        | ThoughtType::Decision
        | ThoughtType::UserTrait
        | ThoughtType::PreferenceUpdate
        | ThoughtType::LessonLearned => HALF_LIFE_LONG_DAYS,
        ThoughtType::Checkpoint | ThoughtType::StateSnapshot => HALF_LIFE_SHORT_DAYS,
        _ => HALF_LIFE_MEDIUM_DAYS,
    }
}

/// Resolve the query-time-decay half-life for a role/type combination,
/// honoring a caller-supplied override before falling back to
/// [`half_life_days`].
pub(crate) fn resolve_half_life_days(
    role: ThoughtRole,
    thought_type: ThoughtType,
    overrides: &HashMap<ThoughtType, u32>,
) -> u32 {
    overrides
        .get(&thought_type)
        .copied()
        .unwrap_or_else(|| half_life_days(role, thought_type))
}

/// Exponential decay ratio in `[0.0, 1.0]` for an age and half-life, both in
/// days. Unfloored: callers that need a floor (query-time decay) apply one
/// separately via [`effective_importance`] or their own `.max(DECAY_FLOOR)`.
pub(crate) fn decay_ratio(age_days: f32, half_life_days: u32) -> f32 {
    let age_days = age_days.max(0.0);
    let half_life = half_life_days.max(1) as f32;
    2f32.powf(-age_days / half_life).clamp(0.0, 1.0)
}

/// Importance after query-time decay, floored at [`DECAY_FLOOR`] so nothing
/// ever reaches exactly zero.
///
/// # Example
///
/// ```
/// use mentisdb::dream::salience::{effective_importance, DECAY_FLOOR};
///
/// assert!(effective_importance(0.9, 0.0, 90) > 0.89);
/// assert!(effective_importance(0.9, 1_000_000.0, 7) >= DECAY_FLOOR);
/// ```
pub fn effective_importance(importance: f32, age_days: f32, half_life_days: u32) -> f32 {
    let importance = importance.clamp(0.0, 1.0);
    DECAY_FLOOR + (importance - DECAY_FLOOR).max(0.0) * decay_ratio(age_days, half_life_days)
}

/// Salience multiplier (`> 1.0`) for `ThoughtType`s the design calls
/// "emotionally salient" analogs: `Surprise`, `Mistake`, `Correction`,
/// `AssumptionInvalidated`, `Decision`, `PreferenceUpdate`, `UserTrait`.
pub(crate) fn type_boost(thought_type: ThoughtType) -> f32 {
    match thought_type {
        ThoughtType::Surprise
        | ThoughtType::Mistake
        | ThoughtType::Correction
        | ThoughtType::AssumptionInvalidated
        | ThoughtType::Decision
        | ThoughtType::PreferenceUpdate
        | ThoughtType::UserTrait => 1.5,
        _ => 1.0,
    }
}

/// Count of raw `refs` and relation entries pointing at `locator`, i.e. the
/// design's literal "refs + relations pointing at the thought" — this counts
/// individual provenance entries, not distinct referring thoughts (two
/// relations from the same source count as 2).
pub(crate) fn in_degree(locator: &ThoughtLocator, adjacency: &ThoughtAdjacencyIndex) -> usize {
    adjacency
        .incoming(locator)
        .iter()
        .map(|edge| edge.provenances.len())
        .sum()
}

/// Effective timestamp for age/decay purposes: the later of a thought's own
/// timestamp and the timestamps of every thought that refs or relates to it
/// ("being consolidated keeps a memory alive," per the design doc).
pub(crate) fn effective_timestamp(
    thought: &Thought,
    locator: &ThoughtLocator,
    adjacency: &ThoughtAdjacencyIndex,
    all_thoughts: &[Thought],
) -> DateTime<Utc> {
    adjacency
        .incoming(locator)
        .iter()
        .filter_map(|edge| edge.source.thought_index)
        .filter_map(|index| all_thoughts.get(index as usize))
        .map(|source| source.timestamp)
        .fold(thought.timestamp, |latest, candidate| latest.max(candidate))
}

/// Salience score used to rank candidates for consolidation and dedup within
/// one dream pass: `importance * decay_ratio(age, half_life) * (1 + ln(1 +
/// in_degree)) * type_boost(type)`.
///
/// Higher-salience thoughts are more likely to be summarized or referenced by
/// a pass. This is a within-pass ranking signal only — it does not persist
/// and is unrelated to [`effective_importance`]'s query-time decay, though
/// both share [`decay_ratio`].
pub(crate) fn salience(
    importance: f32,
    thought_type: ThoughtType,
    age_days: f32,
    half_life_days: u32,
    in_degree: usize,
) -> f32 {
    importance
        * decay_ratio(age_days, half_life_days)
        * (1.0 + (1.0 + in_degree as f32).ln())
        * type_boost(thought_type)
}

#[cfg(test)]
mod tests {
    use crate::search::ThoughtAdjacencyIndex;
    use crate::{
        MentisDb, StorageAdapterKind, ThoughtInput, ThoughtRelation, ThoughtRelationKind,
        ThoughtRole, ThoughtType,
    };
    use tempfile::tempdir;

    fn open_chain() -> (tempfile::TempDir, MentisDb) {
        let dir = tempdir().unwrap();
        let chain = MentisDb::open_with_key_and_storage_kind(
            dir.path(),
            "salience-tests",
            StorageAdapterKind::Binary,
        )
        .unwrap();
        (dir, chain)
    }

    #[test]
    fn type_boost_is_one_point_five_for_emotionally_salient_types() {
        for boosted in [
            ThoughtType::Surprise,
            ThoughtType::Mistake,
            ThoughtType::Correction,
            ThoughtType::AssumptionInvalidated,
            ThoughtType::Decision,
            ThoughtType::PreferenceUpdate,
            ThoughtType::UserTrait,
        ] {
            assert_eq!(
                super::type_boost(boosted),
                1.5,
                "{boosted:?} should be boosted"
            );
        }
    }

    #[test]
    fn type_boost_is_one_for_a_non_boosted_control_type() {
        assert_eq!(super::type_boost(ThoughtType::Idea), 1.0);
    }

    #[test]
    fn half_life_days_is_short_for_working_memory_role_regardless_of_type() {
        // Decision is a Long-tier type, but WorkingMemory role must win.
        let days = super::half_life_days(ThoughtRole::WorkingMemory, ThoughtType::Decision);
        assert_eq!(days, super::HALF_LIFE_SHORT_DAYS);
    }

    #[test]
    fn half_life_days_is_long_for_long_tier_types() {
        for long_type in [
            ThoughtType::Constraint,
            ThoughtType::Decision,
            ThoughtType::UserTrait,
            ThoughtType::PreferenceUpdate,
            ThoughtType::LessonLearned,
        ] {
            assert_eq!(
                super::half_life_days(ThoughtRole::Memory, long_type),
                super::HALF_LIFE_LONG_DAYS,
                "{long_type:?} should be long-tier"
            );
        }
    }

    #[test]
    fn half_life_days_is_short_for_checkpoint_and_state_snapshot_types() {
        assert_eq!(
            super::half_life_days(ThoughtRole::Memory, ThoughtType::Checkpoint),
            super::HALF_LIFE_SHORT_DAYS
        );
        assert_eq!(
            super::half_life_days(ThoughtRole::Memory, ThoughtType::StateSnapshot),
            super::HALF_LIFE_SHORT_DAYS
        );
    }

    #[test]
    fn half_life_days_defaults_unlisted_types_to_medium() {
        // Wonder is one of the 20 types the design doc doesn't explicitly tier.
        assert_eq!(
            super::half_life_days(ThoughtRole::Memory, ThoughtType::Wonder),
            super::HALF_LIFE_MEDIUM_DAYS
        );
    }

    #[test]
    fn resolve_half_life_days_honors_an_override_over_the_table() {
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(ThoughtType::Constraint, 10u32);
        let days =
            super::resolve_half_life_days(ThoughtRole::Memory, ThoughtType::Constraint, &overrides);
        assert_eq!(days, 10);
    }

    #[test]
    fn resolve_half_life_days_falls_back_to_the_table_when_no_override() {
        let overrides = std::collections::HashMap::new();
        let days =
            super::resolve_half_life_days(ThoughtRole::Memory, ThoughtType::Constraint, &overrides);
        assert_eq!(days, super::HALF_LIFE_LONG_DAYS);
    }

    #[test]
    fn effective_importance_is_unchanged_at_zero_age() {
        let value = super::effective_importance(0.8, 0.0, super::HALF_LIFE_MEDIUM_DAYS);
        assert!((value - 0.8).abs() < 1e-6, "expected ~0.8, got {value}");
    }

    #[test]
    fn effective_importance_never_drops_below_the_floor() {
        let value = super::effective_importance(0.9, 1_000_000.0, super::HALF_LIFE_SHORT_DAYS);
        assert!(
            value >= super::DECAY_FLOOR,
            "value {value} dropped below floor"
        );
        assert!(
            (value - super::DECAY_FLOOR).abs() < 1e-6,
            "expected ~floor, got {value}"
        );
    }

    #[test]
    fn salience_decreases_with_age_all_else_fixed() {
        let fresh = super::salience(0.8, ThoughtType::Finding, 0.0, 90, 0);
        let old = super::salience(0.8, ThoughtType::Finding, 365.0, 90, 0);
        assert!(
            old < fresh,
            "old ({old}) should score lower than fresh ({fresh})"
        );
    }

    #[test]
    fn salience_increases_with_in_degree_all_else_fixed() {
        let isolated = super::salience(0.8, ThoughtType::Finding, 1.0, 90, 0);
        let well_referenced = super::salience(0.8, ThoughtType::Finding, 1.0, 90, 5);
        assert!(
            well_referenced > isolated,
            "well-referenced ({well_referenced}) should score higher than isolated ({isolated})"
        );
    }

    #[test]
    fn in_degree_counts_raw_ref_and_relation_entries_not_distinct_sources() {
        let (_dir, mut chain) = open_chain();
        let target = chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, "target thought"),
            )
            .unwrap()
            .clone();
        // One source thought pointing at `target` via TWO separate relations.
        chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Insight, "source thought").with_relations(vec![
                    ThoughtRelation::new(ThoughtRelationKind::RelatedTo, target.id),
                    ThoughtRelation::new(ThoughtRelationKind::References, target.id),
                ]),
            )
            .unwrap();

        let adjacency = ThoughtAdjacencyIndex::from_thoughts(chain.thoughts());
        let locator = crate::search::ThoughtLocator::local(&target);
        assert_eq!(super::in_degree(&locator, &adjacency), 2);
    }

    #[test]
    fn effective_timestamp_is_the_max_of_own_and_inbound_source_timestamps() {
        let (_dir, mut chain) = open_chain();
        let older = chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, "older thought"),
            )
            .unwrap()
            .clone();
        let newer = chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Insight, "newer thought referencing older")
                    .with_relations(vec![ThoughtRelation::new(
                        ThoughtRelationKind::RelatedTo,
                        older.id,
                    )]),
            )
            .unwrap()
            .clone();

        let adjacency = ThoughtAdjacencyIndex::from_thoughts(chain.thoughts());
        let locator = crate::search::ThoughtLocator::local(&older);
        let effective = super::effective_timestamp(&older, &locator, &adjacency, chain.thoughts());
        assert_eq!(effective, older.timestamp.max(newer.timestamp));
    }
}
