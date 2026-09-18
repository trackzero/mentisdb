//! Phase 1 "dreaming" tests: extractive consolidation built on
//! `build_summary_candidates`.
//!
//! See `docs/dreaming-design.md`'s "Phase 1" section for the full design.

use mentisdb::dream::{run_dream_pass, DreamConfig};
use mentisdb::{
    MentisDb, StorageAdapterKind, ThoughtInput, ThoughtRelation, ThoughtRelationKind, ThoughtRole,
    ThoughtType,
};
use tempfile::tempdir;
use uuid::Uuid;

fn open_chain(dir: &std::path::Path, key: &str) -> MentisDb {
    MentisDb::open_with_key_and_storage_kind(dir, key, StorageAdapterKind::Binary).unwrap()
}

fn append(
    chain: &mut MentisDb,
    session: Uuid,
    thought_type: ThoughtType,
    content: &str,
) -> mentisdb::Thought {
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(thought_type, content)
                .with_session_id(session)
                .with_tags(vec![format!("tag-{content}")])
                .with_importance(0.8),
        )
        .unwrap()
        .clone()
}

fn summarizes_targets(chain: &MentisDb) -> Vec<Uuid> {
    chain
        .thoughts()
        .iter()
        .flat_map(|t| t.relations.iter())
        .filter(|r| r.kind == ThoughtRelationKind::Summarizes)
        .map(|r| r.target_id)
        .collect()
}

#[tokio::test]
async fn consolidation_produces_one_summary_per_uncovered_session_window() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "two-windows");

    let session_a = Uuid::new_v4();
    let a1 = append(
        &mut chain,
        session_a,
        ThoughtType::Finding,
        "alpha finding one",
    );
    let a2 = append(
        &mut chain,
        session_a,
        ThoughtType::Finding,
        "alpha finding two",
    );

    let session_b = Uuid::new_v4();
    let b1 = append(
        &mut chain,
        session_b,
        ThoughtType::Finding,
        "beta finding one",
    );
    let b2 = append(
        &mut chain,
        session_b,
        ThoughtType::Finding,
        "beta finding two",
    );

    let before_len = chain.thoughts().len();
    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    assert_eq!(report.counts.consolidations, 2);
    // 2 summaries + 1 report thought.
    assert_eq!(chain.thoughts().len(), before_len + 3);

    let summaries: Vec<_> = chain
        .thoughts()
        .iter()
        .filter(|t| t.role == ThoughtRole::Dream && t.thought_type == ThoughtType::Summary)
        .collect();
    assert_eq!(summaries.len(), 2);

    for summary in &summaries {
        assert!(summary.tags.contains(&"dream".to_string()));
        assert!(summary.tags.contains(&"dream:consolidation".to_string()));
        assert!(summary.confidence.unwrap() <= 0.7);
    }

    let targets = summarizes_targets(&chain);
    for member in [&a1, &a2, &b1, &b2] {
        assert!(
            targets.contains(&member.id),
            "expected a Summarizes relation targeting {}",
            member.id
        );
    }

    // Each summary's tags include the union of ITS members' tags.
    let a_summary = summaries
        .iter()
        .find(|s| {
            s.relations
                .iter()
                .any(|r| r.kind == ThoughtRelationKind::Summarizes && r.target_id == a1.id)
        })
        .unwrap();
    assert!(a_summary
        .tags
        .contains(&"tag-alpha finding one".to_string()));
    assert!(a_summary
        .tags
        .contains(&"tag-alpha finding two".to_string()));
    assert!(!a_summary.tags.contains(&"tag-beta finding one".to_string()));
}

#[tokio::test]
async fn consolidation_skips_a_fully_covered_window() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "covered-window");

    let session_a = Uuid::new_v4();
    let a1 = append(
        &mut chain,
        session_a,
        ThoughtType::Finding,
        "covered finding one",
    );
    let a2 = append(
        &mut chain,
        session_a,
        ThoughtType::Finding,
        "covered finding two",
    );

    // A prior dream-authored summary already covers the whole session-A
    // window. It must not itself be treated as a new candidate (dreams never
    // consolidate other dreams), but its `Summarizes` relations still count
    // as coverage.
    chain
        .append_thought(
            "mentis-dreamer",
            ThoughtInput::new(ThoughtType::Summary, "an earlier dream digest")
                .with_role(ThoughtRole::Dream)
                .with_relations(vec![
                    ThoughtRelation::new(ThoughtRelationKind::Summarizes, a1.id),
                    ThoughtRelation::new(ThoughtRelationKind::Summarizes, a2.id),
                ]),
        )
        .unwrap();

    let session_b = Uuid::new_v4();
    let b1 = append(
        &mut chain,
        session_b,
        ThoughtType::Finding,
        "uncovered finding one",
    );
    let b2 = append(
        &mut chain,
        session_b,
        ThoughtType::Finding,
        "uncovered finding two",
    );

    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    // Only session B's window should be newly consolidated.
    assert_eq!(report.counts.consolidations, 1);
    let targets = summarizes_targets(&chain);
    assert!(targets.contains(&b1.id));
    assert!(targets.contains(&b2.id));
    // a1/a2 are still only targeted by the ORIGINAL covering relations (2),
    // never a second time by a newly generated summary.
    assert_eq!(targets.iter().filter(|id| **id == a1.id).count(), 1);
    assert_eq!(targets.iter().filter(|id| **id == a2.id).count(), 1);
}

#[tokio::test]
async fn consolidation_never_targets_dream_role_or_invalidated_thoughts() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "exclusions");

    let stray_dream = chain
        .append_thought(
            "mentis-dreamer",
            ThoughtInput::new(ThoughtType::Insight, "a stray dream thought")
                .with_role(ThoughtRole::Dream),
        )
        .unwrap()
        .clone();

    let normal = chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Finding, "a normal finding"),
        )
        .unwrap()
        .clone();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Correction, "normal finding was wrong").with_relations(
                vec![ThoughtRelation::new(
                    ThoughtRelationKind::Invalidates,
                    normal.id,
                )],
            ),
        )
        .unwrap();
    assert!(chain.is_invalidated(normal.id));

    run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    let targets = summarizes_targets(&chain);
    assert!(!targets.contains(&stray_dream.id));
    assert!(!targets.contains(&normal.id));
}

#[tokio::test]
async fn dry_run_reports_candidate_consolidations_but_appends_nothing() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dry-run-consolidate");

    let session = Uuid::new_v4();
    append(
        &mut chain,
        session,
        ThoughtType::Finding,
        "dry run finding one",
    );
    append(
        &mut chain,
        session,
        ThoughtType::Finding,
        "dry run finding two",
    );

    let before_len = chain.thoughts().len();
    let report = run_dream_pass(&mut chain, &DreamConfig::default(), true, &[])
        .await
        .unwrap();

    assert_eq!(report.counts.consolidations, 1);
    assert_eq!(chain.thoughts().len(), before_len);
}

#[tokio::test]
async fn consolidation_digest_is_capped_at_2000_chars() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "long-digest");

    let session = Uuid::new_v4();
    let long_statement = "x".repeat(400);
    for i in 0..10 {
        append(
            &mut chain,
            session,
            ThoughtType::Finding,
            &format!("{long_statement}-{i}"),
        );
    }

    run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();

    let summary = chain
        .thoughts()
        .iter()
        .find(|t| t.role == ThoughtRole::Dream && t.thought_type == ThoughtType::Summary)
        .unwrap();
    assert!(summary.content.chars().count() <= 2000);
}
