//! Offline, idle-time consolidation ("dreaming") for MentisDB.
//!
//! Phase 0 provides the scaffolding — role, config, a watermarked pass
//! report, and the trigger surface (MCP/REST/CLI, plus the idle scheduler in
//! the `server`-gated [`scheduler`] submodule). Phase 1 ([`salience`],
//! [`consolidate`], [`suggest`]) adds no-LLM extractive consolidation,
//! query-time decay, and dedup suggestions. Phase 2 ([`recombine`],
//! [`prompts`]) adds LLM-assisted abstractive consolidation, recombination,
//! and contradiction-check, active only when [`DreamConfig::llm`] is
//! configured — the no-LLM core (Phase 0/1 behavior) is otherwise unchanged.
//!
//! See `docs/dreaming-design.md` in the repository for the full design and
//! its non-negotiables: storage stays append-only, the no-LLM core stays
//! intact, every dream write carries provenance, a pass only ever suggests,
//! and everything is off by default.

mod consolidate;
mod prompts;
mod recombine;
pub mod salience;
mod suggest;

#[cfg(feature = "server")]
pub mod scheduler;

use crate::{
    LlmExtractionConfig, LlmExtractionError, MentisDb, ThoughtInput, ThoughtQuery, ThoughtRole,
    ThoughtType, TokenUsage,
};
use async_trait::async_trait;
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
    /// Shared write budget for one pass: extractive/abstractive
    /// consolidation, dedup suggestions, recombination, and contradiction
    /// suggestions all draw from this one pool (in that order), each
    /// decrementing it only on an actual append.
    pub max_writes_per_pass: usize,
    /// Chain key allowlist for the idle scheduler. Empty means the default
    /// chain only.
    pub chains: Vec<String>,
    /// Multiplier applied to a [`ThoughtRole::Dream`] thought's score in
    /// ranked search when it is included via `include_dreams`.
    pub dream_weight: f32,
    /// Optional LLM configuration. `None` keeps the pass fully LLM-free
    /// (Phase 0/1 behavior only). `Some` additionally activates abstractive
    /// consolidation, recombination, and contradiction-check.
    ///
    /// Never serialized: it may carry an API key, and [`DreamConfig::from_env`]
    /// re-reads it from the environment on every call rather than persisting
    /// it. A `DreamConfig` deserialized from JSON always has `llm: None`.
    #[serde(skip)]
    pub llm: Option<crate::LlmExtractionConfig>,
    /// Maximum number of LLM calls per pass across recombination and
    /// contradiction-check combined (a shared pool — there is no separate
    /// contradiction-check budget). Decremented on every call *attempt*,
    /// including ones that end in rejection, so a broken or malicious LLM
    /// can't stall a pass in an unbounded retry loop. Abstractive
    /// consolidation's LLM calls are bounded separately, by
    /// [`Self::max_writes_per_pass`], the same dimension extractive
    /// consolidation already uses.
    pub recombination_budget: usize,
    /// Optional seed for deterministic sampling in later phases.
    ///
    /// Not exercised by any phase implemented so far: extractive/abstractive
    /// consolidation, dedup suggestions, and recombination/contradiction
    /// candidate selection are all deterministic sorts (by salience or
    /// cosine similarity), never a probabilistic sample. Reserved for a
    /// possible future phase.
    pub seed: Option<u64>,
    /// Per-[`crate::ThoughtType`] overrides for the query-time decay half-life
    /// table in [`salience::half_life_days`].
    ///
    /// Defaults to empty, which uses the built-in table unmodified.
    #[serde(default)]
    pub half_life_overrides_days: std::collections::HashMap<crate::ThoughtType, u32>,
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
            half_life_overrides_days: std::collections::HashMap::new(),
        }
    }
}

impl DreamConfig {
    /// Build a [`DreamConfig`] from environment variables, falling back to
    /// [`DreamConfig::default`] for anything unset or unparseable.
    ///
    /// Reads `MENTISDB_DREAM_ENABLED`, `MENTISDB_DREAM_IDLE_SECS`,
    /// `MENTISDB_DREAM_INTERVAL_SECS`, `MENTISDB_DREAM_MAX_SCAN`,
    /// `MENTISDB_DREAM_MAX_WRITES`, `MENTISDB_DREAM_WEIGHT`,
    /// `MENTISDB_DREAM_RECOMBINATION_BUDGET`, and `MENTISDB_DREAM_CHAINS`
    /// (a comma-separated chain-key allowlist for the idle scheduler; empty
    /// or unset means the default chain only, trimmed of surrounding
    /// whitespace per entry).
    ///
    /// `MENTISDB_DREAM_LLM` is a boolean gate, not an LLM config itself: when
    /// it parses as truthy, [`crate::LlmExtractionConfig::from_env`] is
    /// attempted and stored on success; otherwise (including a failed
    /// attempt) `llm` stays `None`, keeping the no-LLM core path the
    /// default.
    ///
    /// `seed` and `half_life_overrides_days` have no env var form: `seed` is
    /// unused by any phase so far (reserved for future sampling), and a
    /// per-type half-life override map has no natural single-variable
    /// encoding — set both via the library API directly if needed.
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            enabled: env_bool("MENTISDB_DREAM_ENABLED").unwrap_or(defaults.enabled),
            idle_after_secs: env_parsed("MENTISDB_DREAM_IDLE_SECS")
                .unwrap_or(defaults.idle_after_secs),
            min_interval_secs: env_parsed("MENTISDB_DREAM_INTERVAL_SECS")
                .unwrap_or(defaults.min_interval_secs),
            max_scan: env_parsed("MENTISDB_DREAM_MAX_SCAN").unwrap_or(defaults.max_scan),
            max_writes_per_pass: env_parsed("MENTISDB_DREAM_MAX_WRITES")
                .unwrap_or(defaults.max_writes_per_pass),
            chains: std::env::var("MENTISDB_DREAM_CHAINS")
                .ok()
                .map(|raw| {
                    raw.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or(defaults.chains),
            dream_weight: env_parsed("MENTISDB_DREAM_WEIGHT").unwrap_or(defaults.dream_weight),
            llm: if env_bool("MENTISDB_DREAM_LLM").unwrap_or(false) {
                crate::LlmExtractionConfig::from_env().ok()
            } else {
                None
            },
            recombination_budget: env_parsed("MENTISDB_DREAM_RECOMBINATION_BUDGET")
                .unwrap_or(defaults.recombination_budget),
            seed: defaults.seed,
            half_life_overrides_days: defaults.half_life_overrides_days,
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
/// `recombinations`/`contradictions`/`rejections` stay `0` whenever
/// `DreamConfig.llm` is `None`, since Phase 2 code never runs in that case.
/// `#[serde(default)]` on each field lets old reports (serialized before a
/// field existed) still deserialize.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DreamPassCounts {
    /// Extractive or abstractive consolidation summaries written.
    #[serde(default)]
    pub consolidations: u64,
    /// Dedup suggestions written.
    #[serde(default)]
    pub suggestions: u64,
    /// Recombination outputs written.
    #[serde(default)]
    pub recombinations: u64,
    /// Contradiction suggestions written.
    #[serde(default)]
    pub contradictions: u64,
    /// LLM outputs rejected by validation (invalid JSON, disallowed type,
    /// over-length content, unresolvable citations, or the novelty filter).
    #[serde(default)]
    pub rejections: u64,
}

/// Structured record of one dream pass.
///
/// Appended as an `Audit`-role `StateSnapshot` thought tagged
/// [`DREAM_REPORT_TAG`] (the report itself is an audit record, not
/// dream-authored content — only a pass's *output*, such as consolidation
/// summaries or recombination suggestions, is [`ThoughtRole::Dream`]). The
/// next pass on the same chain reads the latest report to resume from
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
    /// LLM tokens consumed by this pass across abstractive consolidation,
    /// recombination, and contradiction-check combined. Always `0` when
    /// `DreamConfig.llm` is `None`.
    #[serde(default)]
    pub llm_tokens_used: u64,
    /// Per-operation counters.
    #[serde(default)]
    pub counts: DreamPassCounts,
    /// Phases requested for this pass, validated against
    /// [`DREAM_PHASE_NAMES`]. `"consolidate"` covers extractive/abstractive
    /// consolidation and dedup; `"recombine"` covers recombination and
    /// contradiction-check (only when `DreamConfig.llm` is set); `"decay"`
    /// is a documented no-op here (decay is a query-time ranked-search
    /// factor, not a pass-time operation).
    #[serde(default)]
    pub phases: Vec<String>,
}

/// Seam for the LLM call Phase 2 dreaming makes (abstractive consolidation,
/// recombination, contradiction-check), so tests can inject canned responses
/// instead of hitting a real OpenAI-compatible endpoint.
///
/// `#[async_trait]` is used rather than a hand-rolled boxed future — the same
/// pattern already established for [`crate::server`]'s `ToolProtocol` trait —
/// since `async-trait` is already a dependency and the completer is shared
/// `dyn`-style across multiple call sites within one pass
/// (`dream::recombine`'s recombination and contradiction-check paths, and
/// `dream::consolidate`'s abstractive path).
#[async_trait]
pub(crate) trait ChatCompleter: Send + Sync {
    async fn complete(
        &self,
        config: &LlmExtractionConfig,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<(String, TokenUsage), LlmExtractionError>;
}

/// The real completer used by every production caller: delegates to
/// [`crate::llm::chat_completion`], the shared low-level LLM call helper.
pub(crate) struct RealChatCompleter;

#[async_trait]
impl ChatCompleter for RealChatCompleter {
    async fn complete(
        &self,
        config: &LlmExtractionConfig,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<(String, TokenUsage), LlmExtractionError> {
        crate::llm::chat_completion(config, system, user, temperature).await
    }
}

/// Run one dream pass against `db`.
///
/// Registers the [`DREAM_AGENT_ID`] agent, computes a resumable scan window
/// from the latest pass report (or `max(0, head - max_scan)` on a chain's
/// first pass), and — unless `dry_run` is set — appends a new report. Runs
/// extractive consolidation and dedup suggestions (Phase 1) unconditionally,
/// and, only when [`DreamConfig::llm`] is configured, abstractive
/// consolidation, recombination, and contradiction-check (Phase 2). This
/// function is `async` because Phase 2's LLM calls are; with `config.llm:
/// None` no `.await` point other than the initial call itself is ever
/// reached, and behavior is identical to Phase 0/1.
///
/// Dream-written thoughts are never cryptographically signed: nothing in
/// [`DreamConfig`] supplies key material, and signing elsewhere in this crate
/// is entirely caller-driven (see
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
/// # #[tokio::main]
/// # async fn main() -> std::io::Result<()> {
/// let dir = std::env::temp_dir().join("mentisdb_dream_doctest");
/// let mut chain =
///     MentisDb::open_with_key_and_storage_kind(&dir, "dream-doctest", StorageAdapterKind::Binary)?;
///
/// let report = run_dream_pass(&mut chain, &DreamConfig::default(), true, &[]).await?;
/// assert!(report.dry_run);
/// assert_eq!(chain.thoughts().len(), 0); // dry runs append nothing
/// # let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
pub async fn run_dream_pass(
    db: &mut MentisDb,
    config: &DreamConfig,
    dry_run: bool,
    phases: &[String],
) -> io::Result<DreamReport> {
    run_dream_pass_with_completer(db, config, dry_run, phases, &RealChatCompleter).await
}

/// Test-only (well, `pub(crate)`-only) entry point that lets callers inject a
/// [`ChatCompleter`] instead of the real LLM client. [`run_dream_pass`]
/// itself always calls this with [`RealChatCompleter`]; unit/integration
/// tests call it directly with a canned completer so Phase 2's validation,
/// budget, and novelty-filter logic can be exercised without any real
/// network I/O.
pub(crate) async fn run_dream_pass_with_completer(
    db: &mut MentisDb,
    config: &DreamConfig,
    dry_run: bool,
    phases: &[String],
    completer: &dyn ChatCompleter,
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

    let pass_id = Uuid::new_v4();
    let scan_end_index = db.thoughts().len() as u64;
    let scan_start_index = match latest_report(db) {
        Some(prior) => prior.high_water_index.min(scan_end_index),
        None => scan_end_index.saturating_sub(config.max_scan as u64),
    };
    let scanned_count = scan_end_index.saturating_sub(scan_start_index);

    // "consolidate" covers extractive/abstractive consolidation and dedup
    // suggestions — Phase 1 already bundled dedup under this name since
    // `DREAM_PHASE_NAMES` has no separate entry for it; abstractive
    // consolidation (when `config.llm` is set) is the same phase, just a
    // better digest. "decay" is a documented no-op here: decay is a
    // query-time ranked-search factor (`RankedSearchQuery::use_decay`), not a
    // pass-time operation. "recombine" covers BOTH recombination and
    // contradiction-check (Phase 2's other LLM-assisted work), gated
    // additionally on `config.llm.is_some()` since neither can run without
    // an LLM.
    let run_consolidate = phases.is_empty() || phases.iter().any(|p| p == "consolidate");
    let run_recombine =
        config.llm.is_some() && (phases.is_empty() || phases.iter().any(|p| p == "recombine"));

    let mut write_budget = config.max_writes_per_pass;
    let mut consolidations = Vec::new();
    let mut suggestions = Vec::new();
    let mut consolidation_tokens = 0u64;
    if run_consolidate {
        let llm = config
            .llm
            .as_ref()
            .map(|llm_config| (llm_config, completer));
        let outcome = consolidate::plan_consolidations(
            db,
            config,
            scan_start_index,
            scan_end_index,
            pass_id,
            write_budget,
            llm,
        )
        .await;
        consolidations = outcome.inputs;
        consolidation_tokens = outcome.tokens_used;
        write_budget = write_budget.saturating_sub(consolidations.len());
        suggestions = suggest::plan_dedup_suggestions(
            db,
            config,
            scan_start_index,
            scan_end_index,
            pass_id,
            write_budget,
        );
        write_budget = write_budget.saturating_sub(suggestions.len());
    }

    let mut exploratory = recombine::Phase2ExploratoryOutcome::default();
    if run_recombine {
        // `run_recombine` is only true when `config.llm.is_some()`, so this
        // unwrap is always reached with a real config.
        let llm_config = config
            .llm
            .as_ref()
            .expect("run_recombine implies llm.is_some()");
        exploratory = recombine::plan_recombination_and_contradictions(
            db,
            config,
            llm_config,
            completer,
            scan_start_index,
            scan_end_index,
            pass_id,
            write_budget,
            config.recombination_budget,
        )
        .await;
    }

    let mut report = DreamReport {
        pass_id,
        chain_key,
        started_at,
        duration_ms: 0,
        dry_run,
        scan_start_index,
        scan_end_index,
        high_water_index: scan_end_index,
        scanned_count,
        llm_tokens_used: consolidation_tokens + exploratory.tokens_used,
        counts: DreamPassCounts {
            consolidations: consolidations.len() as u64,
            suggestions: suggestions.len() as u64,
            recombinations: exploratory.recombinations.len() as u64,
            contradictions: exploratory.contradictions.len() as u64,
            rejections: exploratory.rejections,
        },
        phases: phases.to_vec(),
    };

    if dry_run {
        report.duration_ms = started_instant.elapsed().as_millis() as u64;
        return Ok(report);
    }

    for input in consolidations
        .into_iter()
        .chain(suggestions)
        .chain(exploratory.recombinations)
        .chain(exploratory.contradictions)
    {
        db.append_thought(DREAM_AGENT_ID, input)?;
    }

    // This pass is about to append its own report thought, plus whatever
    // consolidation/suggestion/recombination/contradiction thoughts it just
    // wrote above. Advance the watermark past all of them so the next pass's
    // scan window doesn't re-count them as new activity — otherwise a chain
    // with no other appends between passes would never see a truly empty
    // window, and dreams could end up processing their own prior output.
    report.high_water_index = scan_end_index
        + 1
        + report.counts.consolidations
        + report.counts.suggestions
        + report.counts.recombinations
        + report.counts.contradictions;
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
    fn from_env_reads_max_scan_recombination_budget_and_chains() {
        // Unique var names for this test only, to avoid cross-test interference
        // under parallel test execution (no other test in this crate touches
        // these three).
        std::env::set_var("MENTISDB_DREAM_MAX_SCAN", "250");
        std::env::set_var("MENTISDB_DREAM_RECOMBINATION_BUDGET", "7");
        std::env::set_var("MENTISDB_DREAM_CHAINS", "alpha, beta ,gamma");

        let config = DreamConfig::from_env();

        std::env::remove_var("MENTISDB_DREAM_MAX_SCAN");
        std::env::remove_var("MENTISDB_DREAM_RECOMBINATION_BUDGET");
        std::env::remove_var("MENTISDB_DREAM_CHAINS");

        assert_eq!(config.max_scan, 250);
        assert_eq!(config.recombination_budget, 7);
        assert_eq!(config.chains, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn from_env_falls_back_to_defaults_when_new_vars_are_unset() {
        std::env::remove_var("MENTISDB_DREAM_MAX_SCAN");
        std::env::remove_var("MENTISDB_DREAM_RECOMBINATION_BUDGET");
        std::env::remove_var("MENTISDB_DREAM_CHAINS");

        let config = DreamConfig::from_env();

        assert_eq!(config.max_scan, DreamConfig::default().max_scan);
        assert_eq!(
            config.recombination_budget,
            DreamConfig::default().recombination_budget
        );
        assert!(config.chains.is_empty());
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

    // -----------------------------------------------------------------
    // Phase 2: ChatCompleter seam, no-LLM-core invariant, phase gating,
    // budget, and end-to-end wiring. These live here (not in a `tests/*.rs`
    // integration file) because `ChatCompleter`/`run_dream_pass_with_completer`
    // are `pub(crate)` test seams, invisible to external test crates.
    // -----------------------------------------------------------------

    use crate::{StorageAdapterKind, ThoughtInput, ThoughtType};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use tempfile::tempdir;

    fn open_chain(dir: &std::path::Path, key: &str) -> MentisDb {
        MentisDb::open_with_key_and_storage_kind(dir, key, StorageAdapterKind::Binary).unwrap()
    }

    /// Fails the test immediately if invoked. Used to prove `config.llm:
    /// None` means Phase 2 code never constructs or calls a completer.
    struct PanicCompleter;

    #[async_trait]
    impl ChatCompleter for PanicCompleter {
        async fn complete(
            &self,
            _config: &LlmExtractionConfig,
            _system: &str,
            _user: &str,
            _temperature: f32,
        ) -> Result<(String, TokenUsage), LlmExtractionError> {
            panic!("ChatCompleter::complete must never be called when DreamConfig.llm is None");
        }
    }

    /// Returns a canned JSON response keyed by which system persona asked,
    /// and counts total calls made (for budget assertions).
    struct ScriptedCompleter {
        calls: AtomicUsize,
        consolidation_response: Mutex<String>,
        recombination_response: Mutex<String>,
        contradiction_response: Mutex<String>,
    }

    impl ScriptedCompleter {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                consolidation_response: Mutex::new(
                    r#"{"summary": "an llm-rewritten gist"}"#.to_string(),
                ),
                recombination_response: Mutex::new(r#"{"connections": []}"#.to_string()),
                contradiction_response: Mutex::new(r#"{"contradicts": false}"#.to_string()),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ChatCompleter for ScriptedCompleter {
        async fn complete(
            &self,
            _config: &LlmExtractionConfig,
            system: &str,
            _user: &str,
            _temperature: f32,
        ) -> Result<(String, TokenUsage), LlmExtractionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let raw = if system == prompts::CONSOLIDATION_SYSTEM {
                self.consolidation_response.lock().unwrap().clone()
            } else if system == prompts::RECOMBINATION_SYSTEM {
                self.recombination_response.lock().unwrap().clone()
            } else {
                self.contradiction_response.lock().unwrap().clone()
            };
            Ok((raw, TokenUsage::default()))
        }
    }

    fn test_llm_config() -> LlmExtractionConfig {
        LlmExtractionConfig {
            base_url: "http://localhost:0".to_string(),
            api_key: "test-key".to_string(),
            model: "test-model".to_string(),
        }
    }

    fn append_finding(chain: &mut MentisDb, session: Uuid, content: &str) -> crate::Thought {
        chain
            .append_thought(
                "agent",
                ThoughtInput::new(ThoughtType::Finding, content)
                    .with_session_id(session)
                    .with_importance(0.7),
            )
            .unwrap()
            .clone()
    }

    #[tokio::test]
    async fn config_llm_none_never_invokes_the_completer() {
        let dir = tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "no-llm-core");
        let session = Uuid::new_v4();
        append_finding(&mut chain, session, "one finding");
        append_finding(&mut chain, session, "another finding");

        // DreamConfig::default() has llm: None. If any Phase 2 code path
        // were mistakenly reached, PanicCompleter would abort the test.
        let report = run_dream_pass_with_completer(
            &mut chain,
            &DreamConfig::default(),
            false,
            &[],
            &PanicCompleter,
        )
        .await
        .unwrap();

        assert_eq!(report.counts.recombinations, 0);
        assert_eq!(report.counts.contradictions, 0);
        assert_eq!(report.llm_tokens_used, 0);
    }

    #[tokio::test]
    async fn phase_consolidate_alone_does_not_run_recombine_even_with_llm_set() {
        let dir = tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "phase-consolidate-only");
        let session = Uuid::new_v4();
        append_finding(&mut chain, session, "consolidate-only finding one");
        append_finding(&mut chain, session, "consolidate-only finding two");

        let config = DreamConfig {
            llm: Some(test_llm_config()),
            ..DreamConfig::default()
        };
        let completer = ScriptedCompleter::new();
        let report = run_dream_pass_with_completer(
            &mut chain,
            &config,
            false,
            &["consolidate".to_string()],
            &completer,
        )
        .await
        .unwrap();

        assert_eq!(report.counts.recombinations, 0);
        assert_eq!(report.counts.contradictions, 0);
    }

    #[tokio::test]
    async fn phase_recombine_alone_does_not_run_consolidation_or_dedup() {
        let dir = tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "phase-recombine-only");
        let session = Uuid::new_v4();
        append_finding(&mut chain, session, "recombine-only finding one");
        append_finding(&mut chain, session, "recombine-only finding two");

        let config = DreamConfig {
            llm: Some(test_llm_config()),
            ..DreamConfig::default()
        };
        let completer = ScriptedCompleter::new();
        let report = run_dream_pass_with_completer(
            &mut chain,
            &config,
            false,
            &["recombine".to_string()],
            &completer,
        )
        .await
        .unwrap();

        assert_eq!(report.counts.consolidations, 0);
        assert_eq!(report.counts.suggestions, 0);
    }

    #[tokio::test]
    async fn abstractive_consolidation_upgrades_the_digest_and_tags_it() {
        let dir = tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "abstractive-consolidation");
        let session = Uuid::new_v4();
        append_finding(&mut chain, session, "abstractive finding one");
        append_finding(&mut chain, session, "abstractive finding two");

        let config = DreamConfig {
            llm: Some(test_llm_config()),
            ..DreamConfig::default()
        };
        let completer = ScriptedCompleter::new();
        let report = run_dream_pass_with_completer(&mut chain, &config, false, &[], &completer)
            .await
            .unwrap();

        assert_eq!(report.counts.consolidations, 1);
        let summary = chain
            .thoughts()
            .iter()
            .find(|t| t.role == ThoughtRole::Dream && t.thought_type == ThoughtType::Summary)
            .unwrap();
        assert_eq!(summary.content, "an llm-rewritten gist");
        assert!(summary
            .tags
            .contains(&"dream:consolidation:llm".to_string()));
        assert!(summary.confidence.unwrap() <= 0.6);
        assert!(completer.call_count() >= 1);
    }

    #[tokio::test]
    async fn recombination_budget_caps_total_completer_calls() {
        let dir = tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "recombination-budget");
        // Several distinctly-tagged, cross-tagged thoughts so clustering has
        // multiple candidate cluster pairs to consider, exceeding a budget
        // of 1 if left uncapped.
        for i in 0..6 {
            chain
                .append_thought(
                    "agent",
                    ThoughtInput::new(ThoughtType::Finding, format!("budget finding {i}"))
                        .with_tags(vec![format!("tag-{i}")])
                        .with_importance(0.9),
                )
                .unwrap();
        }

        let config = DreamConfig {
            llm: Some(test_llm_config()),
            recombination_budget: 1,
            ..DreamConfig::default()
        };
        let completer = ScriptedCompleter::new();
        run_dream_pass_with_completer(
            &mut chain,
            &config,
            false,
            &["recombine".to_string()],
            &completer,
        )
        .await
        .unwrap();

        assert!(
            completer.call_count() <= 1,
            "expected at most 1 completer call, got {}",
            completer.call_count()
        );
    }

    #[tokio::test]
    async fn chain_integrity_holds_after_a_full_llm_assisted_pass() {
        let dir = tempdir().unwrap();
        let mut chain = open_chain(dir.path(), "integrity-llm");
        let session = Uuid::new_v4();
        append_finding(&mut chain, session, "integrity finding one");
        append_finding(&mut chain, session, "integrity finding two");

        let config = DreamConfig {
            llm: Some(test_llm_config()),
            ..DreamConfig::default()
        };
        let completer = ScriptedCompleter::new();
        run_dream_pass_with_completer(&mut chain, &config, false, &[], &completer)
            .await
            .unwrap();

        assert!(chain.verify_integrity());
    }
}
