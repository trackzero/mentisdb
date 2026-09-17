//! Offline, idle-time consolidation ("dreaming") for MentisDB.
//!
//! Phase 0 provides the scaffolding — role, config, a watermarked pass
//! report, and the trigger surface (MCP/REST/CLI, plus the idle scheduler
//! in the `server`-gated [`scheduler`] submodule) — that later phases
//! (salience sampling, extractive/LLM consolidation, decay, recombination)
//! plug into. [`run_dream_pass`] itself performs no consolidation: it
//! computes a resumable scan window and records a report, nothing else.
//!
//! See `docs/dreaming-design.md` in the repository for the full design and
//! its non-negotiables: storage stays append-only, the no-LLM core stays
//! intact, every dream write carries provenance, a pass only ever suggests,
//! and everything is off by default.

#[cfg(feature = "server")]
pub mod scheduler;

use crate::{MentisDb, ThoughtInput, ThoughtQuery, ThoughtRole, ThoughtType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io;
use uuid::Uuid;

/// Agent id used for every thought a dream pass writes.
pub const DREAM_AGENT_ID: &str = "mentis-dreamer";

/// Tag applied to every dream pass report thought.
pub const DREAM_REPORT_TAG: &str = "dream:report";

/// Names accepted by a dream pass's `phases` parameter.
///
/// Phase 0 validates against this list but has no phase-specific behavior
/// yet — every pass does the same watermark-only scan regardless of which
/// names are given. Phase 1/2 will give each name real meaning.
pub const DREAM_PHASE_NAMES: &[&str] = &["consolidate", "decay", "recombine"];

/// Configuration for offline dream passes.
///
/// Everything defaults to off (`enabled: false`), per MentisDB's "off by
/// default" design bias: nothing in this crate runs a dream pass unless a
/// caller enables the idle scheduler or invokes the manual trigger.
///
/// # Example
///
/// ```
/// use mentisdb::dream::DreamConfig;
///
/// let config = DreamConfig::default();
/// assert!(!config.enabled);
/// assert_eq!(config.idle_after_secs, 900);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Whether the idle scheduler should run at all.
    ///
    /// Manual triggers (the `mentisdb_dream` MCP tool, `POST /v1/dream`, and
    /// the `mentisdb dream` CLI command) ignore this flag and always run
    /// when invoked.
    pub enabled: bool,
    /// Seconds of inactivity (no append by an agent other than
    /// [`DREAM_AGENT_ID`]) before a chain is considered idle.
    pub idle_after_secs: u64,
    /// Minimum seconds between automatic passes on the same chain.
    pub min_interval_secs: u64,
    /// Maximum number of thoughts scanned by a pass that has no prior
    /// watermark to resume from.
    pub max_scan: usize,
    /// Maximum number of thoughts a single pass may append.
    ///
    /// Phase 0 never appends more than the one report thought, so this
    /// bound is not yet exercised; it is validated here so Phase 1/2 have
    /// an enforced budget to plug into.
    pub max_writes_per_pass: usize,
    /// Chain key allowlist for the idle scheduler. Empty means the default
    /// chain only.
    pub chains: Vec<String>,
    /// Multiplier applied to a [`ThoughtRole::Dream`] thought's score in
    /// ranked search when it is included via `include_dreams`.
    pub dream_weight: f32,
    /// Optional LLM configuration for Phase 2 (abstractive consolidation,
    /// recombination). `None` keeps the pass fully LLM-free.
    ///
    /// Never serialized: it may carry an API key, and [`DreamConfig::from_env`]
    /// re-reads it from the environment on every call rather than persisting
    /// it. A `DreamConfig` deserialized from JSON always has `llm: None`.
    #[serde(skip)]
    pub llm: Option<crate::LlmExtractionConfig>,
    /// Maximum number of LLM recombination calls per pass (Phase 2).
    pub recombination_budget: usize,
    /// Optional seed for deterministic sampling in later phases.
    pub seed: Option<u64>,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_after_secs: 900,
            min_interval_secs: 3600,
            max_scan: 500,
            max_writes_per_pass: 20,
            chains: Vec::new(),
            dream_weight: 0.5,
            llm: None,
            recombination_budget: 3,
            seed: None,
        }
    }
}

impl DreamConfig {
    /// Build a [`DreamConfig`] from environment variables, falling back to
    /// [`DreamConfig::default`] for anything unset or unparseable.
    ///
    /// Reads `MENTISDB_DREAM_ENABLED`, `MENTISDB_DREAM_IDLE_SECS`,
    /// `MENTISDB_DREAM_INTERVAL_SECS`, `MENTISDB_DREAM_MAX_WRITES`, and
    /// `MENTISDB_DREAM_WEIGHT`.
    ///
    /// `MENTISDB_DREAM_LLM` is a boolean gate, not an LLM config itself: when
    /// it parses as truthy, [`crate::LlmExtractionConfig::from_env`] is
    /// attempted and stored on success; otherwise (including a failed
    /// attempt) `llm` stays `None`, keeping the no-LLM core path the
    /// default.
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            enabled: env_bool("MENTISDB_DREAM_ENABLED").unwrap_or(defaults.enabled),
            idle_after_secs: env_parsed("MENTISDB_DREAM_IDLE_SECS")
                .unwrap_or(defaults.idle_after_secs),
            min_interval_secs: env_parsed("MENTISDB_DREAM_INTERVAL_SECS")
                .unwrap_or(defaults.min_interval_secs),
            max_scan: defaults.max_scan,
            max_writes_per_pass: env_parsed("MENTISDB_DREAM_MAX_WRITES")
                .unwrap_or(defaults.max_writes_per_pass),
            chains: defaults.chains,
            dream_weight: env_parsed("MENTISDB_DREAM_WEIGHT").unwrap_or(defaults.dream_weight),
            llm: if env_bool("MENTISDB_DREAM_LLM").unwrap_or(false) {
                crate::LlmExtractionConfig::from_env().ok()
            } else {
                None
            },
            recombination_budget: defaults.recombination_budget,
            seed: defaults.seed,
        }
    }
}

fn env_bool(key: &str) -> Option<bool> {
    match std::env::var(key)
        .ok()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "1" | "true" => Some(true),
        "0" | "false" => Some(false),
        _ => None,
    }
}

fn env_parsed<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok()?.trim().parse().ok()
}

/// Per-operation counters for one dream pass.
///
/// Every field is `0` in Phase 0, since no consolidation, decay, or
/// recombination logic exists yet. Later phases populate these as they add
/// behavior; `#[serde(default)]` on each field lets old reports (serialized
/// before a field existed) still deserialize.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DreamPassCounts {
    /// Extractive or abstractive consolidation summaries written.
    #[serde(default)]
    pub consolidations: u64,
    /// Dedup suggestions written.
    #[serde(default)]
    pub suggestions: u64,
    /// Recombination outputs written (Phase 2).
    #[serde(default)]
    pub recombinations: u64,
    /// Contradiction suggestions written (Phase 2).
    #[serde(default)]
    pub contradictions: u64,
    /// LLM outputs rejected by validation (Phase 2).
    #[serde(default)]
    pub rejections: u64,
}

/// Structured record of one dream pass.
///
/// Appended as an `Audit`-role `StateSnapshot` thought tagged
/// [`DREAM_REPORT_TAG`] (the report itself is an audit record, not
/// dream-authored content — only a pass's *output*, such as future
/// consolidation summaries, is [`ThoughtRole::Dream`]). The next pass on the
/// same chain reads the latest report to resume from
/// [`DreamReport::high_water_index`], so the chain stays append-only and
/// needs no side file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamReport {
    /// Unique id of this pass.
    pub pass_id: Uuid,
    /// Chain the pass ran against.
    pub chain_key: String,
    /// When the pass started.
    pub started_at: DateTime<Utc>,
    /// Wall-clock duration of the pass in milliseconds.
    #[serde(default)]
    pub duration_ms: u64,
    /// Whether this was a dry run (nothing appended).
    pub dry_run: bool,
    /// Start of the scanned index range (inclusive): the prior watermark, or
    /// `max(0, head - max_scan)` on a chain's first pass.
    pub scan_start_index: u64,
    /// End of the scanned index range (exclusive): the chain length at pass
    /// start.
    pub scan_end_index: u64,
    /// The index the next pass should resume from.
    pub high_water_index: u64,
    /// Number of thoughts observed in the scan window.
    pub scanned_count: u64,
    /// LLM tokens consumed by this pass (Phase 2; always `0` in Phase 0).
    #[serde(default)]
    pub llm_tokens_used: u64,
    /// Per-operation counters.
    #[serde(default)]
    pub counts: DreamPassCounts,
    /// Phases requested for this pass, validated against
    /// [`DREAM_PHASE_NAMES`] but not yet acted upon.
    #[serde(default)]
    pub phases: Vec<String>,
}

/// Run one dream pass against `db`.
///
/// Registers the [`DREAM_AGENT_ID`] agent, computes a resumable scan window
/// from the latest pass report (or `max(0, head - max_scan)` on a chain's
/// first pass), and — unless `dry_run` is set — appends a new report.
/// Phase 0 performs no consolidation, decay, or recombination: the scan
/// window is bookkeeping for later phases to plug into.
///
/// Dream-written thoughts are never cryptographically signed in Phase 0:
/// nothing in [`DreamConfig`] supplies key material, and signing elsewhere
/// in this crate is entirely caller-driven (see
/// [`crate::ThoughtInput::with_thought_signature`]). Provenance is still
/// fully carried by role, agent id, tags, and relations.
///
/// # Errors
///
/// Returns an error if any entry in `phases` is not one of
/// [`DREAM_PHASE_NAMES`], or if appending the report thought fails.
///
/// # Example
///
/// ```
/// use mentisdb::dream::{run_dream_pass, DreamConfig};
/// use mentisdb::{MentisDb, StorageAdapterKind};
///
/// # fn main() -> std::io::Result<()> {
/// let dir = std::env::temp_dir().join("mentisdb_dream_doctest");
/// let mut chain =
///     MentisDb::open_with_key_and_storage_kind(&dir, "dream-doctest", StorageAdapterKind::Binary)?;
///
/// let report = run_dream_pass(&mut chain, &DreamConfig::default(), true, &[])?;
/// assert!(report.dry_run);
/// assert_eq!(chain.thoughts().len(), 0); // dry runs append nothing
/// # let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
pub fn run_dream_pass(
    db: &mut MentisDb,
    config: &DreamConfig,
    dry_run: bool,
    phases: &[String],
) -> io::Result<DreamReport> {
    for phase in phases {
        if !DREAM_PHASE_NAMES.contains(&phase.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown dream phase '{phase}'"),
            ));
        }
    }

    let started_at = Utc::now();
    let started_instant = std::time::Instant::now();
    let chain_key = db.chain_key().to_string();

    db.upsert_agent(
        DREAM_AGENT_ID,
        Some(DREAM_AGENT_ID),
        None,
        Some("Automated offline consolidation pass (see docs/dreaming-design.md)"),
        None,
    )?;

    let scan_end_index = db.thoughts().len() as u64;
    let scan_start_index = match latest_report(db) {
        Some(prior) => prior.high_water_index.min(scan_end_index),
        None => scan_end_index.saturating_sub(config.max_scan as u64),
    };
    let scanned_count = scan_end_index.saturating_sub(scan_start_index);

    let mut report = DreamReport {
        pass_id: Uuid::new_v4(),
        chain_key,
        started_at,
        duration_ms: 0,
        dry_run,
        scan_start_index,
        scan_end_index,
        high_water_index: scan_end_index,
        scanned_count,
        llm_tokens_used: 0,
        counts: DreamPassCounts::default(),
        phases: phases.to_vec(),
    };

    if dry_run {
        report.duration_ms = started_instant.elapsed().as_millis() as u64;
        return Ok(report);
    }

    // This pass is about to append exactly one thought (its own report).
    // Advance the watermark past it so the next pass's scan window doesn't
    // re-count it as new activity — otherwise a chain with no other appends
    // between passes would never see a truly empty window.
    report.high_water_index = scan_end_index + 1;
    report.duration_ms = started_instant.elapsed().as_millis() as u64;
    let content = serde_json::to_string(&report)
        .map_err(|e| io::Error::other(format!("failed to serialize dream report: {e}")))?;
    let input = ThoughtInput::new(ThoughtType::StateSnapshot, content)
        .with_role(ThoughtRole::Audit)
        .with_tags([
            DREAM_REPORT_TAG.to_string(),
            format!("dream:pass:{}", report.pass_id),
        ]);
    db.append_thought(DREAM_AGENT_ID, input)?;

    #[cfg(feature = "server")]
    {
        if let Some(webhook_manager) = db.webhook_manager.clone() {
            let chain_key = report.chain_key.clone();
            let report_for_webhook = report.clone();
            tokio::spawn(async move {
                webhook_manager.deliver_dream_report(&chain_key, &report_for_webhook);
            });
        }
    }

    Ok(report)
}

/// Find the most recently appended dream pass report, if any.
fn latest_report(db: &MentisDb) -> Option<DreamReport> {
    let query = ThoughtQuery::new()
        .with_types(vec![ThoughtType::StateSnapshot])
        .with_roles(vec![ThoughtRole::Audit])
        .with_tags_any([DREAM_REPORT_TAG])
        .with_agent_ids([DREAM_AGENT_ID])
        .with_include_invalidated(true);
    db.query(&query)
        .into_iter()
        .max_by_key(|thought| thought.index)
        .and_then(|thought| serde_json::from_str(&thought.content).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_design_doc_defaults() {
        let config = DreamConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.idle_after_secs, 900);
        assert_eq!(config.min_interval_secs, 3600);
        assert_eq!(config.max_scan, 500);
        assert_eq!(config.max_writes_per_pass, 20);
        assert!(config.chains.is_empty());
        assert_eq!(config.dream_weight, 0.5);
        assert!(config.llm.is_none());
        assert_eq!(config.recombination_budget, 3);
        assert_eq!(config.seed, None);
    }

    #[test]
    fn dream_pass_counts_round_trip_all_zero() {
        let counts = DreamPassCounts::default();
        let json = serde_json::to_string(&counts).unwrap();
        let restored: DreamPassCounts = serde_json::from_str(&json).unwrap();
        assert_eq!(counts, restored);
        assert_eq!(restored.consolidations, 0);
        assert_eq!(restored.rejections, 0);
    }

    #[test]
    fn dream_pass_counts_deserializes_from_empty_object_for_forward_compat() {
        let restored: DreamPassCounts = serde_json::from_str("{}").unwrap();
        assert_eq!(restored, DreamPassCounts::default());
    }
}
