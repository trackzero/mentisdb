//! Idle-time trigger for dream passes.
//!
//! Runs a coarse `tokio::time::interval` loop that checks each configured
//! chain for idleness and, when idle, runs a dream pass via
//! [`crate::server::MentisDbService::dream`] — the same entry point (and the
//! same per-chain overlap lock) used by the manual MCP/REST trigger, so an
//! automatic pass and a manual one can never race on one chain.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::oneshot;

use super::{DreamConfig, DREAM_AGENT_ID};
use crate::server::{DreamRequest, MentisDbService};
use crate::MentisDb;

/// Coarse poll interval for the idle scheduler. Not user-configurable in
/// Phase 0 — the design only calls for "a coarse interval".
const DEFAULT_TICK: Duration = Duration::from_secs(30);

/// A handle to a running idle dream scheduler.
///
/// Returned by [`spawn_dream_scheduler`]. Dropping the handle without
/// calling [`shutdown`](Self::shutdown) leaves the scheduler running in the
/// background; call `shutdown` for a clean stop.
#[derive(Debug)]
pub struct DreamSchedulerHandle {
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl DreamSchedulerHandle {
    /// Signal the scheduler loop to stop after its current tick.
    ///
    /// Calling this a second time is a no-op that returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns an error if the shutdown signal cannot be delivered because
    /// the scheduler's background task has already exited.
    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(tx) = self.shutdown_tx.take() {
            tx.send(())
                .map_err(|_| "dream scheduler shutdown signal could not be delivered".into())
        } else {
            Ok(())
        }
    }
}

/// Spawn the idle dream scheduler with the default coarse tick interval.
///
/// Does nothing on its own unless `config.enabled` is `true` — callers
/// (namely [`crate::server::start_servers`]) are expected to check that
/// before calling this. Iterates `config.chains` (or just the service's
/// default chain when empty, per the design) every tick and runs a dream
/// pass on each chain that is idle and past `min_interval_secs` since its
/// last pass.
pub fn spawn_dream_scheduler(
    service: Arc<MentisDbService>,
    config: DreamConfig,
) -> DreamSchedulerHandle {
    spawn_dream_scheduler_with_tick(service, config, DEFAULT_TICK)
}

/// Like [`spawn_dream_scheduler`], but with an explicit tick interval.
///
/// Exposed mainly so tests (and advanced tuning) don't have to wait on the
/// production default of 30 seconds.
pub fn spawn_dream_scheduler_with_tick(
    service: Arc<MentisDbService>,
    config: DreamConfig,
    tick: Duration,
) -> DreamSchedulerHandle {
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(tick);
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    run_tick(&service, &config).await;
                }
                _ = &mut shutdown_rx => break,
            }
        }
    });
    DreamSchedulerHandle {
        shutdown_tx: Some(shutdown_tx),
    }
}

async fn run_tick(service: &Arc<MentisDbService>, config: &DreamConfig) {
    let chain_keys: Vec<String> = if config.chains.is_empty() {
        vec![service.resolve_chain_key(None)]
    } else {
        config.chains.clone()
    };

    for chain_key in chain_keys {
        maybe_run_pass(service, config, &chain_key).await;
    }
}

async fn maybe_run_pass(service: &Arc<MentisDbService>, config: &DreamConfig, chain_key: &str) {
    let chain = match service.get_chain(Some(chain_key), None).await {
        Ok(chain) => chain,
        Err(_) => return, // Skip chains that fail to open; don't stop the loop.
    };

    let is_idle = {
        let guard = chain.read().await;
        chain_is_idle(&guard, config)
    };
    if !is_idle {
        return;
    }

    let _ = service
        .dream(DreamRequest {
            chain_key: Some(chain_key.to_string()),
            dry_run: Some(false),
            phases: None,
        })
        .await;
}

/// A chain is idle when there has been no append by any agent other than
/// [`DREAM_AGENT_ID`] for `idle_after_secs`, and the last dream pass (if
/// any) was at least `min_interval_secs` ago.
///
/// This scans from the tail of the chain backwards and stops at the first
/// non-dreamer thought, so it is O(1) in the common case where the most
/// recent thought is not dreamer-authored. It only degenerates to a fuller
/// scan when the tail is a long run of dream-authored thoughts, which is
/// acceptable at a coarse tick since `max_scan` already bounds real pass
/// cost — a Phase 0 simplification, not a permanent index.
fn chain_is_idle(db: &MentisDb, config: &DreamConfig) -> bool {
    let now = Utc::now();

    let last_other_agent_activity = db
        .thoughts()
        .iter()
        .rev()
        .find(|thought| thought.agent_id != DREAM_AGENT_ID)
        .map(|thought| thought.timestamp);
    let idle_long_enough = match last_other_agent_activity {
        Some(timestamp) => (now - timestamp).num_seconds() >= config.idle_after_secs as i64,
        None => true, // No non-dreamer activity at all: treat as idle.
    };
    if !idle_long_enough {
        return false;
    }

    match super::latest_report(db) {
        Some(report) => (now - report.started_at).num_seconds() >= config.min_interval_secs as i64,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StorageAdapterKind, ThoughtInput, ThoughtType};
    use chrono::Duration as ChronoDuration;

    fn open_chain(dir: &std::path::Path, key: &str) -> MentisDb {
        MentisDb::open_with_key_and_storage_kind(dir, key, StorageAdapterKind::Binary).unwrap()
    }

    #[test]
    fn a_chain_with_no_thoughts_is_idle() {
        let dir = tempfile::tempdir().unwrap();
        let chain = open_chain(dir.path(), "empty");
        assert!(chain_is_idle(&chain, &DreamConfig::default()));
    }

    #[test]
    fn a_chain_with_a_very_recent_append_is_not_idle() {
        let dir = tempfile::tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "recent");
        chain
            .append_thought("agent", ThoughtInput::new(ThoughtType::Insight, "just now"))
            .unwrap();
        let config = DreamConfig {
            idle_after_secs: 3600,
            ..DreamConfig::default()
        };
        assert!(!chain_is_idle(&chain, &config));
    }

    #[test]
    fn a_chain_whose_only_recent_activity_is_the_dreamer_is_idle() {
        let dir = tempfile::tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "dreamer-only");
        chain
            .append_thought(
                DREAM_AGENT_ID,
                ThoughtInput::new(ThoughtType::StateSnapshot, "report"),
            )
            .unwrap();
        let config = DreamConfig {
            idle_after_secs: 3600,
            ..DreamConfig::default()
        };
        assert!(chain_is_idle(&chain, &config));
    }

    #[test]
    fn min_interval_since_the_last_report_is_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "min-interval");
        let old_report = super::super::DreamReport {
            pass_id: uuid::Uuid::new_v4(),
            chain_key: "min-interval".to_string(),
            started_at: Utc::now() - ChronoDuration::seconds(10),
            duration_ms: 0,
            dry_run: false,
            scan_start_index: 0,
            scan_end_index: 0,
            high_water_index: 0,
            scanned_count: 0,
            llm_tokens_used: 0,
            counts: super::super::DreamPassCounts::default(),
            phases: vec![],
        };
        chain
            .append_thought(
                DREAM_AGENT_ID,
                ThoughtInput::new(
                    ThoughtType::StateSnapshot,
                    serde_json::to_string(&old_report).unwrap(),
                )
                .with_role(crate::ThoughtRole::Audit)
                .with_tags(["dream:report".to_string()]),
            )
            .unwrap();

        let strict_config = DreamConfig {
            idle_after_secs: 0,
            min_interval_secs: 3600,
            ..DreamConfig::default()
        };
        assert!(!chain_is_idle(&chain, &strict_config));

        let lenient_config = DreamConfig {
            idle_after_secs: 0,
            min_interval_secs: 1,
            ..DreamConfig::default()
        };
        assert!(chain_is_idle(&chain, &lenient_config));
    }
}
