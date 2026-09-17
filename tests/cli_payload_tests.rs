//! Payload-shape regression tests for the `mentisdb add` and `mentisdb search`
//! CLI subcommands.
//!
//! GitHub issue #17 reported that these subcommands produced REST payloads the
//! daemon rejected because the required `thought_type` field was missing. These
//! tests pin the outgoing JSON body so the CLI and the REST schema cannot drift
//! again without a failing test.

use mentisdb::cli::{
    build_add_body, build_dream_body, build_ranked_search_body, parse_args, AddCommand, CliCommand,
    DreamCommand, SearchCommand,
};

fn add_command(content: &str, thought_type: Option<&str>) -> AddCommand {
    AddCommand {
        content: content.to_string(),
        thought_type: thought_type.map(str::to_string),
        scope: None,
        tags: Vec::new(),
        agent_id: None,
        chain_key: None,
        url: "http://127.0.0.1:9472".to_string(),
    }
}

fn search_command(text: &str) -> SearchCommand {
    SearchCommand {
        text: text.to_string(),
        limit: None,
        scope: None,
        chain_key: None,
        url: "http://127.0.0.1:9472".to_string(),
    }
}

/// The REST API requires `thought_type`, so omitting `--type` must still emit
/// the documented default rather than an empty body field.
#[test]
fn add_body_defaults_thought_type_to_fact_learned() {
    let body = build_add_body(&add_command("smoke test", None)).unwrap();
    assert_eq!(body["thought_type"], "FactLearned");
    assert_eq!(body["content"], "smoke test");
}

/// Explicit types are normalized to their canonical PascalCase name.
#[test]
fn add_body_normalizes_explicit_thought_type() {
    assert_eq!(
        build_add_body(&add_command("x", Some("insight"))).unwrap()["thought_type"],
        "Insight"
    );
    assert_eq!(
        build_add_body(&add_command("x", Some("fact-learned"))).unwrap()["thought_type"],
        "FactLearned"
    );
    // `Goal` was missing from the server's old hand-written parser; the unified
    // parser accepts it and the CLI forwards it.
    assert_eq!(
        build_add_body(&add_command("x", Some("goal"))).unwrap()["thought_type"],
        "Goal"
    );
}

/// A typo in `--type` fails locally with an actionable message instead of an
/// opaque server 4xx.
#[test]
fn add_body_rejects_unknown_thought_type_with_valid_options() {
    let error = build_add_body(&add_command("x", Some("note"))).unwrap_err();
    assert!(
        error.contains("Unknown ThoughtType 'note'"),
        "unexpected error: {error}"
    );
    assert!(
        error.contains("FactLearned"),
        "error should list valid types"
    );
    assert!(error.contains("Insight"), "error should list valid types");
}

/// Optional flags map onto their REST field names.
#[test]
fn add_body_carries_optional_fields() {
    let mut cmd = add_command("x", Some("decision"));
    cmd.scope = Some("session".to_string());
    cmd.tags = vec!["a".to_string(), "b".to_string()];
    cmd.agent_id = Some("agent-1".to_string());
    cmd.chain_key = Some("chain-1".to_string());

    let body = build_add_body(&cmd).unwrap();
    assert_eq!(body["scope"], "session");
    assert_eq!(body["tags"], serde_json::json!(["a", "b"]));
    assert_eq!(body["agent_id"], "agent-1");
    assert_eq!(body["chain_key"], "chain-1");
}

/// Ranked search reads the query from `text`, not `query`.
#[test]
fn search_body_uses_text_field() {
    let mut cmd = search_command("cache invalidation");
    cmd.limit = Some(5);
    cmd.scope = Some("user".to_string());
    cmd.chain_key = Some("chain-1".to_string());

    let body = build_ranked_search_body(&cmd);
    assert_eq!(body["text"], "cache invalidation");
    assert_eq!(body["limit"], 5);
    assert_eq!(body["scope"], "user");
    assert_eq!(body["chain_key"], "chain-1");
    assert!(
        body.get("query").is_none(),
        "must not send the old query field"
    );
}

fn dream_command(chain: Option<&str>, dry_run: bool, phase: Option<Vec<&str>>) -> DreamCommand {
    DreamCommand {
        chain: chain.map(str::to_string),
        dry_run,
        phase: phase.map(|p| p.into_iter().map(str::to_string).collect()),
        url: "http://127.0.0.1:9472".to_string(),
    }
}

/// The dream body maps CLI fields onto the `POST /v1/dream` request shape.
#[test]
fn dream_body_maps_fields() {
    let body = build_dream_body(&dream_command(
        Some("chain-1"),
        true,
        Some(vec!["consolidate", "decay"]),
    ));
    assert_eq!(body["chain_key"], "chain-1");
    assert_eq!(body["dry_run"], true);
    assert_eq!(body["phases"], serde_json::json!(["consolidate", "decay"]));
}

/// Omitted optional fields are left out of the body rather than sent as null,
/// matching the other subcommands' payload style.
#[test]
fn dream_body_omits_absent_optional_fields() {
    let body = build_dream_body(&dream_command(None, false, None));
    assert!(body.get("chain_key").is_none());
    assert_eq!(body["dry_run"], false);
    assert!(body.get("phases").is_none());
}

/// `mentisdb dream --dry-run` parses with no chain/phase set.
#[test]
fn parse_args_parses_dream_dry_run() {
    let command = parse_args(["mentisdb", "dream", "--dry-run"]).unwrap();
    assert_eq!(
        command,
        CliCommand::Dream(DreamCommand {
            chain: None,
            dry_run: true,
            phase: None,
            url: "http://127.0.0.1:9472".to_string(),
        })
    );
}

/// `--chain` and a comma-separated `--phase` list both parse.
#[test]
fn parse_args_parses_dream_chain_and_phase_list() {
    let command = parse_args([
        "mentisdb",
        "dream",
        "--chain",
        "project-x",
        "--phase",
        "consolidate,decay",
    ])
    .unwrap();
    assert_eq!(
        command,
        CliCommand::Dream(DreamCommand {
            chain: Some("project-x".to_string()),
            dry_run: false,
            phase: Some(vec!["consolidate".to_string(), "decay".to_string()]),
            url: "http://127.0.0.1:9472".to_string(),
        })
    );
}

/// An unknown phase name fails locally with an actionable message instead of
/// an opaque server 4xx.
#[test]
fn parse_args_rejects_unknown_dream_phase() {
    let error = parse_args(["mentisdb", "dream", "--phase", "not-a-real-phase"]).unwrap_err();
    assert!(
        error.contains("not-a-real-phase"),
        "unexpected error: {error}"
    );
}
