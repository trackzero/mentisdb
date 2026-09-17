#![cfg(feature = "server")]
//! Phase 0 "dreaming" server-surface tests: `include_dreams` on
//! `recent_context`, the `mentisdb_dream` MCP tool, and the
//! `POST /v1/dream` REST endpoint.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mentisdb::server::{mcp_router, rest_router, MentisDbServiceConfig};
use mentisdb::StorageAdapterKind;
use serde_json::json;
use tower::util::ServiceExt;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_chain_dir() -> PathBuf {
    let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "mentisdb_dream_server_test_{}_{}",
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn append_thought(
    router: axum::Router,
    chain_key: &str,
    agent_id: &str,
    thought_type: &str,
    role: Option<&str>,
    content: &str,
) -> serde_json::Value {
    let mut payload = json!({
        "chain_key": chain_key,
        "agent_id": agent_id,
        "thought_type": thought_type,
        "content": content
    });
    if let Some(role) = role {
        payload["role"] = json!(role);
    }
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/thoughts")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    body_json(response).await
}

#[tokio::test]
async fn recent_context_excludes_dream_role_by_default_and_includes_when_requested() {
    let dir = unique_chain_dir();
    let chain_key = "recent-context-dreams";
    let router = rest_router(MentisDbServiceConfig::new(
        dir.clone(),
        chain_key,
        StorageAdapterKind::Binary,
    ));

    append_thought(
        router.clone(),
        chain_key,
        "agent",
        "Insight",
        None,
        "a normal memory",
    )
    .await;
    append_thought(
        router.clone(),
        chain_key,
        "mentis-dreamer",
        "Insight",
        Some("Dream"),
        "a dream digest",
    )
    .await;

    let default_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/recent-context")
                .header("content-type", "application/json")
                .body(Body::from(json!({ "chain_key": chain_key }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(default_response.status(), StatusCode::OK);
    let default_json = body_json(default_response).await;
    let default_prompt = default_json["prompt"].as_str().unwrap();
    assert!(default_prompt.contains("a normal memory"));
    assert!(!default_prompt.contains("a dream digest"));

    let with_dreams_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/recent-context")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "chain_key": chain_key, "include_dreams": true }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let with_dreams_json = body_json(with_dreams_response).await;
    let with_dreams_prompt = with_dreams_json["prompt"].as_str().unwrap();
    assert!(with_dreams_prompt.contains("a normal memory"));
    assert!(with_dreams_prompt.contains("a dream digest"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn mcp_dream_tool_appears_in_tool_list() {
    let dir = unique_chain_dir();
    let router = mcp_router(MentisDbServiceConfig::new(
        dir.clone(),
        "dream-tool-list",
        StorageAdapterKind::Binary,
    ));

    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tools/list")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    let tools = json["tools"].as_array().unwrap();
    assert!(tools.iter().any(|tool| tool["name"] == "mentisdb_dream"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn mcp_dream_tool_dry_run_then_real_trigger() {
    let dir = unique_chain_dir();
    let chain_key = "mcp-dream-trigger";
    let router = mcp_router(MentisDbServiceConfig::new(
        dir.clone(),
        chain_key,
        StorageAdapterKind::Binary,
    ));

    let dry_run = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tools/execute")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "tool": "mentisdb_dream",
                        "parameters": { "chain_key": chain_key, "dry_run": true }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(dry_run.status(), StatusCode::OK);
    let dry_run_json = body_json(dry_run).await;
    assert_eq!(dry_run_json["result"]["success"], json!(true));
    assert_eq!(
        dry_run_json["result"]["output"]["report"]["dry_run"],
        json!(true)
    );

    let real_run = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tools/execute")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "tool": "mentisdb_dream",
                        "parameters": { "chain_key": chain_key }
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(real_run.status(), StatusCode::OK);
    let real_run_json = body_json(real_run).await;
    assert_eq!(
        real_run_json["result"]["output"]["report"]["dry_run"],
        json!(false)
    );

    let chain = mentisdb::MentisDb::open_with_key_and_storage_kind(
        &dir,
        chain_key,
        StorageAdapterKind::Binary,
    )
    .unwrap();
    assert_eq!(chain.thoughts().len(), 1);
    assert_eq!(chain.thoughts()[0].agent_id, "mentis-dreamer");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn rest_dream_endpoint_dry_run_and_manual_trigger() {
    let dir = unique_chain_dir();
    let chain_key = "rest-dream-trigger";
    let router = rest_router(MentisDbServiceConfig::new(
        dir.clone(),
        chain_key,
        StorageAdapterKind::Binary,
    ));

    let dry_run = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/dream")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({ "chain_key": chain_key, "dry_run": true }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(dry_run.status(), StatusCode::OK);
    let dry_run_json = body_json(dry_run).await;
    assert_eq!(dry_run_json["report"]["dry_run"], json!(true));

    let real_run = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/dream")
                .header("content-type", "application/json")
                .body(Body::from(json!({ "chain_key": chain_key }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(real_run.status(), StatusCode::OK);
    let real_run_json = body_json(real_run).await;
    assert_eq!(real_run_json["report"]["dry_run"], json!(false));

    let _ = std::fs::remove_dir_all(&dir);
}
