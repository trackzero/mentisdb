//! Phase 1 "dreaming" tests: dedup suggestions.
//!
//! See `docs/dreaming-design.md`'s "Phase 1" section, "Dedup suggestions".

use mentisdb::dream::{run_dream_pass, DreamConfig};
use mentisdb::search::{EmbeddingInput, EmbeddingMetadata, EmbeddingProvider, EmbeddingVector};
use mentisdb::{
    MentisDb, StorageAdapterKind, ThoughtInput, ThoughtRelation, ThoughtRelationKind, ThoughtRole,
    ThoughtType,
};
use std::convert::Infallible;
use tempfile::tempdir;

/// Deterministic test provider: the FIRST WORD of the embedded text picks a
/// fixed direction, so tests control cosine similarity independently of the
/// real wording used for lexical (Jaccard) overlap.
#[derive(Clone)]
struct BucketProvider {
    metadata: EmbeddingMetadata,
}

impl BucketProvider {
    fn new() -> Self {
        Self {
            metadata: EmbeddingMetadata::new("dream-suggest-test", 2, "v1"),
        }
    }

    fn vector_for_text(text: &str) -> Vec<f32> {
        if text.contains("vecbucketalpha") {
            vec![1.0, 0.0]
        } else if text.contains("vecbucketbeta") {
            vec![0.0, 1.0]
        } else if text.contains("vecbucketgamma") {
            vec![-1.0, 0.0]
        } else {
            vec![0.2, 0.2]
        }
    }
}

impl EmbeddingProvider for BucketProvider {
    type Error = Infallible;

    fn metadata(&self) -> &EmbeddingMetadata {
        &self.metadata
    }

    fn embed_batch(&self, inputs: &[EmbeddingInput]) -> Result<Vec<EmbeddingVector>, Self::Error> {
        Ok(inputs
            .iter()
            .map(|input| EmbeddingVector::new(Self::vector_for_text(&input.text)))
            .collect())
    }
}

fn open_chain(dir: &std::path::Path, key: &str) -> MentisDb {
    MentisDb::open_with_key_and_storage_kind(dir, key, StorageAdapterKind::Binary).unwrap()
}

fn no_supersedes_invalidates_or_corrects(chain: &MentisDb) -> bool {
    chain.thoughts().iter().all(|t| {
        t.relations.iter().all(|r| {
            !matches!(
                r.kind,
                ThoughtRelationKind::Supersedes
                    | ThoughtRelationKind::Invalidates
                    | ThoughtRelationKind::Corrects
            )
        })
    })
}

fn suggestion_thoughts(chain: &MentisDb) -> Vec<&mentisdb::Thought> {
    chain
        .thoughts()
        .iter()
        .filter(|t| {
            t.role == ThoughtRole::Dream && t.tags.contains(&"dream:suggestion".to_string())
        })
        .collect()
}

#[tokio::test]
async fn near_duplicate_same_agent_pair_with_sidecar_produces_one_suggestion() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dedup-positive");

    let a = chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance",
            ),
        )
        .unwrap()
        .clone();
    let b = chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance now",
            ),
        )
        .unwrap()
        .clone();

    chain.manage_vector_sidecar(BucketProvider::new()).unwrap();

    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    assert_eq!(report.counts.suggestions, 1);

    let suggestions = suggestion_thoughts(&chain);
    assert_eq!(suggestions.len(), 1);
    let suggestion = suggestions[0];
    assert_eq!(suggestion.thought_type, ThoughtType::Finding);
    let related: Vec<_> = suggestion
        .relations
        .iter()
        .filter(|r| r.kind == ThoughtRelationKind::RelatedTo)
        .map(|r| r.target_id)
        .collect();
    assert_eq!(related.len(), 2);
    assert!(related.contains(&a.id));
    assert!(related.contains(&b.id));

    assert!(no_supersedes_invalidates_or_corrects(&chain));
}

#[tokio::test]
async fn same_pair_from_different_agents_produces_no_suggestion() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dedup-cross-agent");

    chain
        .append_thought(
            "agent-a",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance",
            ),
        )
        .unwrap();
    chain
        .append_thought(
            "agent-b",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance now",
            ),
        )
        .unwrap();

    chain.manage_vector_sidecar(BucketProvider::new()).unwrap();

    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    assert_eq!(report.counts.suggestions, 0);
    assert!(suggestion_thoughts(&chain).is_empty());
}

#[tokio::test]
async fn high_cosine_but_low_lexical_overlap_produces_no_suggestion() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dedup-lexical-gate");

    chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha the quick brown fox jumps over the lazy dog near the riverbank",
            ),
        )
        .unwrap();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha completely unrelated sentence about oceanography and marine biology",
            ),
        )
        .unwrap();

    chain.manage_vector_sidecar(BucketProvider::new()).unwrap();

    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    assert_eq!(report.counts.suggestions, 0);
}

#[tokio::test]
async fn high_lexical_overlap_but_low_cosine_produces_no_suggestion() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dedup-cosine-gate");

    chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketbeta runbook step rotate api keys quarterly for security compliance",
            ),
        )
        .unwrap();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketgamma runbook step rotate api keys quarterly for security compliance",
            ),
        )
        .unwrap();

    chain.manage_vector_sidecar(BucketProvider::new()).unwrap();

    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    assert_eq!(report.counts.suggestions, 0);
}

#[tokio::test]
async fn already_related_pair_is_skipped() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dedup-already-linked");

    let a = chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance",
            ),
        )
        .unwrap()
        .clone();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance now",
            )
            .with_relations(vec![ThoughtRelation::new(
                ThoughtRelationKind::RelatedTo,
                a.id,
            )]),
        )
        .unwrap();

    chain.manage_vector_sidecar(BucketProvider::new()).unwrap();

    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    assert_eq!(report.counts.suggestions, 0);
}

#[tokio::test]
async fn no_vector_sidecar_configured_skips_dedup_cleanly() {
    let dir = tempdir().unwrap();
    let mut chain = open_chain(dir.path(), "dedup-no-sidecar");

    chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance",
            ),
        )
        .unwrap();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(
                ThoughtType::Finding,
                "vecbucketalpha runbook step rotate api keys quarterly for security compliance now",
            ),
        )
        .unwrap();

    // No manage_vector_sidecar call: no embedding space is configured.
    let report = run_dream_pass(&mut chain, &DreamConfig::default(), false, &[])
        .await
        .unwrap();
    assert_eq!(report.counts.suggestions, 0);
}
