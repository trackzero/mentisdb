//! Phase 3 "dreaming" tests: `promote_dream`/`dismiss_dream` helpers.
//!
//! See `docs/dreaming-design.md`'s "Promotion (\"waking up\")" subsection.

use mentisdb::{
    MentisDb, StorageAdapterKind, ThoughtInput, ThoughtRelationKind, ThoughtRole, ThoughtType,
};
use tempfile::tempdir;

fn open_chain(dir: &std::path::Path, key: &str) -> MentisDb {
    MentisDb::open_with_key_and_storage_kind(dir, key, StorageAdapterKind::Binary).unwrap()
}

fn append_dream(chain: &mut MentisDb, content: &str) -> mentisdb::Thought {
    chain
        .append_thought(
            "mentis-dreamer",
            ThoughtInput::new(ThoughtType::Finding, content).with_role(ThoughtRole::Dream),
        )
        .unwrap()
        .clone()
}

#[test]
fn promote_dream_appends_memory_thought_with_derived_from() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "promote-basic");
    let dream = append_dream(&mut chain, "a dream finding");

    let promoted = chain.promote_dream("reviewer", dream.id, None).unwrap();

    assert_eq!(promoted.role, ThoughtRole::Memory);
    // Promotion keeps the dream's own semantic type (e.g. Finding stays a
    // Finding) -- only the role changes from Dream to Memory. `ThoughtType`
    // has no generic "Memory" variant; role is what "Memory-role" means.
    assert_eq!(promoted.thought_type, ThoughtType::Finding);
    assert_eq!(promoted.agent_id, "reviewer");
    assert!(promoted
        .relations
        .iter()
        .any(|r| r.kind == ThoughtRelationKind::DerivedFrom && r.target_id == dream.id));
}

#[test]
fn promote_dream_without_edited_content_copies_dream_content_verbatim() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "promote-verbatim");
    let dream = append_dream(&mut chain, "the original dream text");

    let promoted = chain.promote_dream("reviewer", dream.id, None).unwrap();

    assert_eq!(promoted.content, "the original dream text");
}

#[test]
fn promote_dream_with_edited_content_uses_the_override() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "promote-edited");
    let dream = append_dream(&mut chain, "the original dream text");

    let promoted = chain
        .promote_dream("reviewer", dream.id, Some("a cleaned-up version"))
        .unwrap();

    assert_eq!(promoted.content, "a cleaned-up version");
}

#[test]
fn promote_dream_errors_on_nonexistent_id() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "promote-missing");

    let err = chain
        .promote_dream("reviewer", uuid::Uuid::new_v4(), None)
        .unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn promote_dream_errors_on_non_dream_role_thought() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "promote-wrong-role");
    let normal = chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Finding, "a normal finding"),
        )
        .unwrap()
        .clone();

    let err = chain
        .promote_dream("reviewer", normal.id, None)
        .unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn dismiss_dream_appends_audit_correction_with_invalidates() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dismiss-basic");
    let dream = append_dream(&mut chain, "a dream suggestion");

    let dismissal = chain
        .dismiss_dream("reviewer", dream.id, Some("not useful"))
        .unwrap();

    assert_eq!(dismissal.role, ThoughtRole::Audit);
    assert_eq!(dismissal.thought_type, ThoughtType::Correction);
    assert_eq!(dismissal.content, "not useful");
    assert!(dismissal
        .relations
        .iter()
        .any(|r| r.kind == ThoughtRelationKind::Invalidates && r.target_id == dream.id));
}

#[test]
fn dismiss_dream_marks_the_dream_invalidated_immediately() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dismiss-invalidates");
    let dream = append_dream(&mut chain, "a dream suggestion");

    chain.dismiss_dream("reviewer", dream.id, None).unwrap();

    assert!(chain.is_invalidated(dream.id));
}

#[test]
fn dismiss_dream_without_reason_uses_placeholder_content() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dismiss-no-reason");
    let dream = append_dream(&mut chain, "a dream suggestion");

    let dismissal = chain.dismiss_dream("reviewer", dream.id, None).unwrap();

    assert_eq!(dismissal.content, "Dismissed without a stated reason.");
}

#[test]
fn dismiss_dream_errors_on_nonexistent_id() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dismiss-missing");

    let err = chain
        .dismiss_dream("reviewer", uuid::Uuid::new_v4(), None)
        .unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn dismiss_dream_errors_on_non_dream_role_thought() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dismiss-wrong-role");
    let normal = chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Finding, "a normal finding"),
        )
        .unwrap()
        .clone();

    let err = chain
        .dismiss_dream("reviewer", normal.id, None)
        .unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn promote_and_dismiss_can_both_be_applied_to_the_same_dream() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "promote-then-dismiss");
    let dream = append_dream(&mut chain, "a contested dream");

    chain.promote_dream("reviewer-a", dream.id, None).unwrap();
    chain
        .dismiss_dream("reviewer-b", dream.id, Some("actually wrong"))
        .unwrap();

    let derived_from_count = chain
        .thoughts()
        .iter()
        .flat_map(|t| t.relations.iter())
        .filter(|r| r.kind == ThoughtRelationKind::DerivedFrom && r.target_id == dream.id)
        .count();
    let invalidates_count = chain
        .thoughts()
        .iter()
        .flat_map(|t| t.relations.iter())
        .filter(|r| r.kind == ThoughtRelationKind::Invalidates && r.target_id == dream.id)
        .count();
    assert_eq!(derived_from_count, 1);
    assert_eq!(invalidates_count, 1);
}
