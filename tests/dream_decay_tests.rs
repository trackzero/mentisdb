//! Phase 1 "dreaming" tests: query-time decay (`RankedSearchQuery::use_decay`).
//!
//! See `docs/dreaming-design.md`'s "Phase 1" section, "Decay (query-time
//! only)". The actual decay math (exponential curve, floor, half-life
//! precedence, age-refresh via inbound references) is unit-tested directly
//! in `src/dream/salience.rs`, since it operates on synthetic `age_days`
//! values. These tests instead cover the `RankedSearchQuery`/`query_ranked`
//! wiring: the flag defaults to off, is a true no-op when off, and doesn't
//! misbehave (crash, spuriously drop fresh results) when turned on.
//!
//! Note: there is no way to backdate a thought's `timestamp` through the
//! public API (it is always stamped at append time), so a true "an old
//! thought ranks lower than a fresh one" integration fixture isn't
//! constructible here — that property is covered at the unit level in
//! `src/dream/salience.rs`'s `salience_decreases_with_age` and
//! `effective_importance_never_drops_below_the_floor` tests instead.

use mentisdb::dream::DreamConfig;
use mentisdb::{
    MentisDb, RankedSearchQuery, StorageAdapterKind, ThoughtInput, ThoughtRole, ThoughtType,
};
use tempfile::tempdir;

fn open_chain(dir: &std::path::Path, key: &str) -> MentisDb {
    MentisDb::open_with_key_and_storage_kind(dir, key, StorageAdapterKind::Binary).unwrap()
}

#[test]
fn use_decay_defaults_to_false() {
    let query = RankedSearchQuery::new();
    assert!(!query.use_decay);
}

#[test]
fn explicit_use_decay_false_is_byte_identical_to_unset() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "decay-off-identical");
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Finding, "decay wiring text").with_importance(0.6),
        )
        .unwrap();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Constraint, "decay wiring text").with_importance(0.9),
        )
        .unwrap();

    let unset = chain.query_ranked(&RankedSearchQuery::new().with_text("decay wiring text"));
    let explicit_off = chain.query_ranked(
        &RankedSearchQuery::new()
            .with_text("decay wiring text")
            .with_use_decay(false),
    );

    assert_eq!(unset.hits.len(), explicit_off.hits.len());
    for (a, b) in unset.hits.iter().zip(explicit_off.hits.iter()) {
        assert_eq!(a.thought.id, b.thought.id);
        assert_eq!(a.score.total, b.score.total);
    }
}

#[test]
fn use_decay_true_does_not_change_the_candidate_set_for_equally_fresh_thoughts() {
    // All thoughts here are appended back-to-back, so their effective age is
    // ~0 for every half-life tier: decay_ratio(~0, half_life) ~= 1.0
    // regardless of type/role. Enabling decay must not spuriously drop or
    // reorder equally-fresh results relative to it being off.
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "decay-on-fresh");
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Finding, "fresh decay candidate one")
                .with_importance(0.6),
        )
        .unwrap();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Constraint, "fresh decay candidate two")
                .with_importance(0.9),
        )
        .unwrap();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Idea, "fresh decay candidate three")
                .with_role(ThoughtRole::WorkingMemory)
                .with_importance(0.8),
        )
        .unwrap();

    let off = chain.query_ranked(&RankedSearchQuery::new().with_text("fresh decay candidate"));
    let on = chain.query_ranked(
        &RankedSearchQuery::new()
            .with_text("fresh decay candidate")
            .with_use_decay(true),
    );

    let off_ids: Vec<_> = off.hits.iter().map(|h| h.thought.id).collect();
    let on_ids: Vec<_> = on.hits.iter().map(|h| h.thought.id).collect();
    assert_eq!(
        off_ids, on_ids,
        "decay must not reorder equally-fresh candidates"
    );
}

#[test]
fn use_decay_true_with_half_life_overrides_does_not_panic() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "decay-overrides");
    chain.with_dream_config(DreamConfig {
        half_life_overrides_days: std::collections::HashMap::from([(ThoughtType::Finding, 1)]),
        ..DreamConfig::default()
    });
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Finding, "override plumbing text").with_importance(0.7),
        )
        .unwrap();

    let result = chain.query_ranked(
        &RankedSearchQuery::new()
            .with_text("override plumbing text")
            .with_use_decay(true),
    );
    assert_eq!(result.hits.len(), 1);
}
