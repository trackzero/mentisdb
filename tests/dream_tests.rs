//! Phase 0 "dreaming" tests: `ThoughtRole::Dream` plumbing, `include_dreams`,
//! `DreamConfig`, and the watermarked `run_dream_pass` scaffolding.
//!
//! See `docs/dreaming-design.md` for the full design. Phase 0 performs no
//! consolidation, decay, or recombination — these tests cover the
//! scaffolding those later phases will plug into.

use mentisdb::dream::{run_dream_pass, DreamConfig, DreamPassCounts};
use mentisdb::{
    MentisDb, RankedSearchQuery, StorageAdapterKind, ThoughtInput, ThoughtQuery, ThoughtRole,
    ThoughtType,
};
use tempfile::tempdir;
use uuid::Uuid;

fn open_chain(dir: &std::path::Path, key: &str) -> MentisDb {
    MentisDb::open_with_key_and_storage_kind(dir, key, StorageAdapterKind::Binary).unwrap()
}

fn append_role(chain: &mut MentisDb, role: ThoughtRole, content: &str) -> mentisdb::Thought {
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Insight, content)
                .with_role(role)
                .with_importance(0.9),
        )
        .unwrap()
        .clone()
}

// ---------------------------------------------------------------------
// ThoughtRole::Dream plumbing
// ---------------------------------------------------------------------

#[test]
fn dream_role_serializes_and_deserializes() {
    let json = serde_json::to_string(&ThoughtRole::Dream).unwrap();
    let restored: ThoughtRole = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ThoughtRole::Dream);
}

#[test]
fn dream_role_round_trips_through_memory_markdown_import() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "role-roundtrip");

    let markdown =
        "## Findings\n\n- [#0] Finding: consolidated gist (agent mentis-dreamer; role Dream; importance 0.60)\n";
    let indices = chain
        .import_from_memory_markdown(markdown, "default-agent")
        .unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(chain.thoughts()[0].role, ThoughtRole::Dream);
}

#[test]
fn a_chain_written_before_the_dream_role_existed_still_opens() {
    // The role enum has no serde rename, so an old chain that never used the
    // new Dream variant should be unaffected: appending and reopening a
    // chain that only ever used pre-existing roles must still work exactly
    // as before.
    let dir = tempdir().unwrap();
    {
        let mut chain = open_chain(dir.path(), "pre-existing");
        append_role(&mut chain, ThoughtRole::Memory, "unrelated old memory");
    }
    let reopened = open_chain(dir.path(), "pre-existing");
    assert_eq!(reopened.thoughts().len(), 1);
    assert_eq!(reopened.thoughts()[0].role, ThoughtRole::Memory);
}

// ---------------------------------------------------------------------
// include_dreams
// ---------------------------------------------------------------------

#[test]
fn default_query_excludes_dream_role_thoughts() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "query-dreams");
    append_role(&mut chain, ThoughtRole::Memory, "a normal memory");
    append_role(&mut chain, ThoughtRole::Dream, "a dream digest");

    let default_hits = chain.query(&ThoughtQuery::new());
    assert_eq!(default_hits.len(), 1);
    assert_eq!(default_hits[0].role, ThoughtRole::Memory);

    let with_dreams = chain.query(&ThoughtQuery::new().with_include_dreams(true));
    assert_eq!(with_dreams.len(), 2);
}

#[test]
fn default_ranked_search_excludes_dream_role_thoughts() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "ranked-dreams");
    append_role(&mut chain, ThoughtRole::Memory, "shared search text");
    append_role(&mut chain, ThoughtRole::Dream, "shared search text");

    let default_result =
        chain.query_ranked(&RankedSearchQuery::new().with_text("shared search text"));
    assert_eq!(default_result.hits.len(), 1);
    assert_eq!(default_result.hits[0].thought.role, ThoughtRole::Memory);

    let with_dreams = chain.query_ranked(
        &RankedSearchQuery::new()
            .with_text("shared search text")
            .with_include_dreams(true),
    );
    assert_eq!(with_dreams.hits.len(), 2);
}

#[test]
fn included_dream_hits_are_down_weighted_by_dream_weight() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "ranked-dream-weight");
    chain.with_dream_config(DreamConfig {
        dream_weight: 0.1,
        ..DreamConfig::default()
    });
    append_role(&mut chain, ThoughtRole::Memory, "weighted search text");
    append_role(&mut chain, ThoughtRole::Dream, "weighted search text");

    let result = chain.query_ranked(
        &RankedSearchQuery::new()
            .with_text("weighted search text")
            .with_include_dreams(true),
    );
    assert_eq!(result.hits.len(), 2);
    let memory_score = result
        .hits
        .iter()
        .find(|hit| hit.thought.role == ThoughtRole::Memory)
        .unwrap()
        .score
        .total;
    let dream_score = result
        .hits
        .iter()
        .find(|hit| hit.thought.role == ThoughtRole::Dream)
        .unwrap()
        .score
        .total;
    assert!(
        dream_score < memory_score,
        "dream hit ({dream_score}) should score lower than the equivalent memory hit ({memory_score})"
    );
}

#[test]
fn memory_markdown_never_includes_dream_role_thoughts() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "markdown-dreams");
    append_role(&mut chain, ThoughtRole::Memory, "visible memory content");
    append_role(&mut chain, ThoughtRole::Dream, "hidden dream content");

    let full_dump = chain.to_memory_markdown(None);
    assert!(full_dump.contains("visible memory content"));
    assert!(!full_dump.contains("hidden dream content"));

    let queried = chain.to_memory_markdown(Some(&ThoughtQuery::new()));
    assert!(!queried.contains("hidden dream content"));
}

// ---------------------------------------------------------------------
// DreamConfig
// ---------------------------------------------------------------------

#[test]
fn dream_config_default_is_off() {
    let config = DreamConfig::default();
    assert!(!config.enabled);
}

// ---------------------------------------------------------------------
// run_dream_pass: watermark, dry_run, budgets, provenance
// ---------------------------------------------------------------------

#[tokio::test]
async fn dry_run_appends_nothing_but_reflects_the_would_be_scan() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dry-run");
    append_role(&mut chain, ThoughtRole::Memory, "some memory");

    let before_len = chain.thoughts().len();
    let report = run_dream_pass(&mut chain, &DreamConfig::default(), true, &[])
        .await
        .unwrap();

    assert_eq!(chain.thoughts().len(), before_len);
    assert!(report.dry_run);
    assert_eq!(report.scan_end_index, before_len as u64);
    assert_eq!(report.scanned_count, before_len as u64);
}

#[tokio::test]
async fn a_real_pass_appends_exactly_one_report_thought_and_registers_the_dreamer() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "real-pass");
    append_role(&mut chain, ThoughtRole::Memory, "some memory");

    let before_len = chain.thoughts().len();
    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    assert!(!report.dry_run);
    assert_eq!(chain.thoughts().len(), before_len + 1);
    let appended = chain.thoughts().last().unwrap();
    assert_eq!(appended.thought_type, ThoughtType::StateSnapshot);
    assert_eq!(appended.role, ThoughtRole::Audit);
    assert!(appended.tags.contains(&"dream:report".to_string()));
    assert_eq!(appended.agent_id, "mentis-dreamer");

    assert!(chain
        .list_agent_registry()
        .iter()
        .any(|agent| agent.agent_id == "mentis-dreamer"));
}

#[tokio::test]
async fn a_second_immediate_pass_scans_an_empty_window() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "second-pass");
    append_role(&mut chain, ThoughtRole::Memory, "some memory");

    let first = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    let after_first_len = chain.thoughts().len();

    let second = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    assert_eq!(second.scan_start_index, first.high_water_index);
    assert_eq!(second.scan_start_index, second.scan_end_index);
    assert_eq!(second.scanned_count, 0);
    // Only the new report thought was appended, nothing else.
    assert_eq!(chain.thoughts().len(), after_first_len + 1);
}

#[tokio::test]
async fn first_pass_on_a_long_chain_starts_from_head_minus_max_scan() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "long-chain");
    for i in 0..10 {
        append_role(&mut chain, ThoughtRole::Memory, &format!("memory {i}"));
    }
    let head = chain.thoughts().len() as u64;
    let config = DreamConfig {
        max_scan: 3,
        ..DreamConfig::default()
    };

    let report = run_dream_pass(&mut chain, &config, true, &[])
        .await
        .unwrap();

    assert_eq!(report.scan_start_index, head - 3);
    assert_eq!(report.scan_end_index, head);
}

#[tokio::test]
async fn first_pass_on_a_short_chain_starts_from_zero() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "short-chain");
    append_role(&mut chain, ThoughtRole::Memory, "one memory");
    let config = DreamConfig {
        max_scan: 500,
        ..DreamConfig::default()
    };

    let report = run_dream_pass(&mut chain, &config, true, &[])
        .await
        .unwrap();

    assert_eq!(report.scan_start_index, 0);
}

#[tokio::test]
async fn unknown_phase_names_are_rejected() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "bad-phase");
    let result = run_dream_pass(
        &mut chain,
        &DreamConfig::default(),
        true,
        &["not-a-real-phase".to_string()],
    )
    .await;
    assert!(result.is_err());
    assert_eq!(chain.thoughts().len(), 0);
}

#[tokio::test]
async fn known_phase_names_are_accepted_and_echoed_back() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "good-phase");
    let report = run_dream_pass(
        &mut chain,
        &DreamConfig::default(),
        true,
        &["consolidate".to_string(), "decay".to_string()],
    )
    .await
    .unwrap();
    assert_eq!(report.phases, vec!["consolidate", "decay"]);
}

#[tokio::test]
async fn max_writes_per_pass_of_zero_does_not_panic() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "zero-budget");
    let config = DreamConfig {
        max_writes_per_pass: 0,
        ..DreamConfig::default()
    };
    let report = run_dream_pass(&mut chain, &config, false, &[])
        .await
        .unwrap();
    assert_eq!(report.counts, DreamPassCounts::default());
}

#[tokio::test]
async fn dream_pass_does_not_append_supersedes_invalidates_or_corrects() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "suggest-only");
    append_role(&mut chain, ThoughtRole::Memory, "existing memory");

    run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    for thought in chain.thoughts() {
        for relation in &thought.relations {
            assert!(!matches!(
                relation.kind,
                mentisdb::ThoughtRelationKind::Supersedes
                    | mentisdb::ThoughtRelationKind::Invalidates
                    | mentisdb::ThoughtRelationKind::Corrects
            ));
        }
    }
}

#[tokio::test]
async fn chain_integrity_holds_after_a_dream_pass() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "integrity");
    append_role(&mut chain, ThoughtRole::Memory, "some memory");

    run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    assert!(chain.verify_integrity());
}

// ---------------------------------------------------------------------
// Phase 1: combined budget, real-write integrity, decay-phase no-op
// ---------------------------------------------------------------------

#[tokio::test]
async fn max_writes_per_pass_caps_combined_consolidation_and_dedup_writes() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "combined-budget");
    // Five separate session-grouped windows, each independently
    // consolidatable, comfortably exceeding a budget of 2.
    for session in 0..5 {
        let session_id = Uuid::new_v4();
        for i in 0..2 {
            chain
                .append_thought(
                    "agent",
                    ThoughtInput::new(
                        ThoughtType::Finding,
                        format!("session {session} finding {i}"),
                    )
                    .with_session_id(session_id)
                    .with_importance(0.7),
                )
                .unwrap();
        }
    }

    let config = DreamConfig {
        max_writes_per_pass: 2,
        ..DreamConfig::default()
    };
    let report = run_dream_pass(&mut chain, &config, false, &[])
        .await
        .unwrap();

    let total_writes = report.counts.consolidations + report.counts.suggestions;
    assert!(
        total_writes <= 2,
        "expected at most 2 combined writes, got {total_writes}"
    );
}

#[tokio::test]
async fn chain_integrity_holds_after_a_pass_with_real_consolidation_writes() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "integrity-with-writes");
    let session = Uuid::new_v4();
    for i in 0..3 {
        chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, format!("integrity finding {i}"))
                    .with_session_id(session)
                    .with_importance(0.7),
            )
            .unwrap();
    }

    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    assert!(
        report.counts.consolidations > 0,
        "expected at least one consolidation to exercise integrity"
    );

    assert!(chain.verify_integrity());
}

#[tokio::test]
async fn phases_decay_only_performs_watermark_bookkeeping() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "decay-phase-noop");
    let session = Uuid::new_v4();
    for i in 0..2 {
        chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, format!("decay-phase finding {i}"))
                    .with_session_id(session)
                    .with_importance(0.7),
            )
            .unwrap();
    }

    let before_len = chain.thoughts().len();
    let report = run_dream_pass(
        &mut chain,
        &DreamConfig::default(),
        false,
        &["decay".to_string()],
    )
    .await
    .unwrap();

    assert_eq!(report.counts, DreamPassCounts::default());
    // Only the report thought is appended: no consolidation/dedup ran.
    assert_eq!(chain.thoughts().len(), before_len + 1);
}
