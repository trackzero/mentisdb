#![cfg(feature = "server")]
//! Phase 0 "dreaming" idle scheduler tests: idle detection, chain
//! allowlisting, and clean shutdown.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use mentisdb::dream::scheduler::spawn_dream_scheduler_with_tick;
use mentisdb::dream::DreamConfig;
use mentisdb::server::{MentisDbService, MentisDbServiceConfig};
use mentisdb::{MentisDb, StorageAdapterKind, ThoughtInput, ThoughtType};
use std::sync::Arc;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_chain_dir() -> PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "mentisdb_dream_scheduler_test_{}_{}",
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn seed_chain(dir: &std::path::Path, chain_key: &str) {
    let mut chain =
        MentisDb::open_with_key_and_storage_kind(dir, chain_key, StorageAdapterKind::Binary)
            .unwrap();
    chain
        .append_thought(
            "agent",
            ThoughtInput::new(ThoughtType::Insight, "seed thought"),
        )
        .unwrap();
}

fn report_count(dir: &std::path::Path, chain_key: &str) -> usize {
    let chain =
        MentisDb::open_with_key_and_storage_kind(dir, chain_key, StorageAdapterKind::Binary)
            .unwrap();
    chain
        .thoughts()
        .iter()
        .filter(|t| t.tags.contains(&"dream:report".to_string()))
        .count()
}

async fn wait_until(mut condition: impl FnMut() -> bool, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    condition()
}

#[tokio::test]
async fn scheduler_appends_a_report_for_an_idle_chain() {
    let dir = unique_chain_dir();
    let chain_key = "idle-chain";
    seed_chain(&dir, chain_key);

    let service = Arc::new(MentisDbService::new(MentisDbServiceConfig::new(
        dir.clone(),
        chain_key,
        StorageAdapterKind::Binary,
    )));
    let config = DreamConfig {
        idle_after_secs: 0,
        min_interval_secs: 0,
        chains: vec![chain_key.to_string()],
        ..DreamConfig::default()
    };
    let mut handle = spawn_dream_scheduler_with_tick(service, config, Duration::from_millis(30));

    let appeared = wait_until(
        || report_count(&dir, chain_key) >= 1,
        Duration::from_secs(2),
    )
    .await;
    assert!(
        appeared,
        "expected a dream report to appear for the idle chain"
    );

    handle.shutdown().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn scheduler_skips_a_chain_that_is_not_idle() {
    let dir = unique_chain_dir();
    let chain_key = "busy-chain";
    seed_chain(&dir, chain_key);

    let service = Arc::new(MentisDbService::new(MentisDbServiceConfig::new(
        dir.clone(),
        chain_key,
        StorageAdapterKind::Binary,
    )));
    let config = DreamConfig {
        idle_after_secs: 3600,
        min_interval_secs: 0,
        chains: vec![chain_key.to_string()],
        ..DreamConfig::default()
    };
    let mut handle = spawn_dream_scheduler_with_tick(service, config, Duration::from_millis(30));

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        report_count(&dir, chain_key),
        0,
        "a recently-active chain must not get an automatic dream pass"
    );

    handle.shutdown().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn scheduler_only_runs_allowlisted_chains() {
    let dir = unique_chain_dir();
    let allowed = "allowed-chain";
    let other = "other-chain";
    seed_chain(&dir, allowed);
    seed_chain(&dir, other);

    let service = Arc::new(MentisDbService::new(MentisDbServiceConfig::new(
        dir.clone(),
        allowed,
        StorageAdapterKind::Binary,
    )));
    let config = DreamConfig {
        idle_after_secs: 0,
        min_interval_secs: 0,
        chains: vec![allowed.to_string()],
        ..DreamConfig::default()
    };
    let mut handle = spawn_dream_scheduler_with_tick(service, config, Duration::from_millis(30));

    let appeared = wait_until(|| report_count(&dir, allowed) >= 1, Duration::from_secs(2)).await;
    assert!(appeared, "the allowlisted chain should get a dream pass");

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        report_count(&dir, other),
        0,
        "a chain outside the allowlist must never run automatically"
    );

    handle.shutdown().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn shutdown_stops_the_loop_promptly() {
    let dir = unique_chain_dir();
    let chain_key = "shutdown-chain";
    seed_chain(&dir, chain_key);

    let service = Arc::new(MentisDbService::new(MentisDbServiceConfig::new(
        dir.clone(),
        chain_key,
        StorageAdapterKind::Binary,
    )));
    let config = DreamConfig {
        idle_after_secs: 0,
        min_interval_secs: 0,
        chains: vec![chain_key.to_string()],
        ..DreamConfig::default()
    };
    let mut handle = spawn_dream_scheduler_with_tick(service, config, Duration::from_millis(20));

    let appeared = wait_until(
        || report_count(&dir, chain_key) >= 1,
        Duration::from_secs(2),
    )
    .await;
    assert!(appeared);

    handle.shutdown().unwrap();
    let count_at_shutdown = report_count(&dir, chain_key);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        report_count(&dir, chain_key),
        count_at_shutdown,
        "no further passes should run after shutdown"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
