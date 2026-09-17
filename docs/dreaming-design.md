# Dreaming: Offline Consolidation for MentisDB

Status: design, not implemented. Branch: `feature/dreaming`.

## Why

Deployed agent harnesses only touch memory when a request arrives. Biological
memory does a lot of its work offline: sleep replays recent experience, folds it
into durable gist, renormalizes what got over-strengthened during the day, and
loosely recombines distant memories (one proposed source of insight). Dreaming
gives MentisDB an idle-time pass that does the same jobs:

| Biological function | MentisDB analog | Phase |
|---|---|---|
| Salience-biased replay (sharp-wave ripples) | Salience sampler picks what each pass touches | 1 |
| Systems consolidation (episodic → gist) | Summaries over uncovered clusters | 1 (extractive), 2 (LLM) |
| Synaptic homeostasis (downscaling) | Query-time decay by type, age and in-degree; dedup suggestions | 1 |
| Associative recombination (REM) | LLM links between distant, salient clusters | 2 |

The goal is dreaming as *function*, not dreaming as *content*. A job that just
generates surreal text and stores it as fact would slowly poison the store with
things nobody said. Every design choice below is there to prevent that.

## Non-negotiables

These follow from `AGENTS.md` and the design bias of the crate.

1. **Append-only.** Dreaming never deletes or rewrites. Decay is a ranking
   factor computed at query time. Pruning means ranking something down, never
   removing it.
2. **No-LLM core.** Phases 0 and 1 work with no LLM configured. Phase 2 sits
   behind the existing opt-in LLM path (`llm.rs`, `LlmExtractionConfig`).
3. **Provenance on everything.** Every thought a dream pass writes is
   identifiable as a dream (see below) and links to its sources.
4. **Suggest, don't act.** A dream pass never appends `Supersedes`,
   `Invalidates`, or `Corrects` on its own authority. It records suggestions,
   and an awake agent or a human confirms them.
5. **Off by default.** Nothing runs unless it is configured.
6. **LLM output is untrusted.** The same review and validation posture as
   `extract_memories`.
7. Public API gets rustdoc, and new behavior gets tests. Integrity checks and
   signing still apply to dream-written thoughts.

## Core concepts

### Dream provenance

- Add `ThoughtRole::Dream`, with rustdoc: "Written by an unprompted offline
  consolidation or recombination pass; lower trust until promoted." Update every
  role parser, docs, the skill file and the dashboard filters.
- Dream thoughts are written by a registered agent, `mentis-dreamer`, with a
  description, and signed if keys are configured.
- Tags: `dream`, `dream:pass:<uuid>`, plus `dream:consolidation` /
  `dream:recombination` / `dream:suggestion`.
- Confidence caps: extractive consolidation ≤ 0.7, LLM consolidation ≤ 0.6,
  recombination ≤ 0.4.
- Relations: `Summarizes` for consolidation, `DerivedFrom` (one per source) for
  recombination, `RelatedTo` for suggestions.

### Retrieval treatment

- Follow the existing `include_invalidated` pattern: add `include_dreams: bool`
  to the ranked-search and recent-context query types, and to the MCP, REST and
  CLI surfaces.
- `recent_context` and `memory_markdown` leave dreams out by default.
- `ranked_search` includes dreams but multiplies their score by
  `dream_weight` (default 0.5, configurable). Each result is labeled as a dream
  in its output.

### Promotion ("waking up")

An agent or a human promotes a dream by appending a normal `Memory`-role
thought with `DerivedFrom → dream`. They dismiss one by appending an
`Audit`-role thought with `Invalidates → dream`. Both are plain appends, so no
new storage semantics are needed. Expose them as the helpers
`promote_dream(id, edited_content?)` and `dismiss_dream(id, reason)` across the
library, MCP, REST and CLI surfaces.

### Pass report and watermark

Each pass ends by appending an `Audit`-role `StateSnapshot` tagged
`dream:report`. Its content is a structured summary: the pass id, the chain,
the index range scanned, counts per operation, the LLM tokens used and the
duration. It also records the **high-water index**. The next pass reads the
latest report to resume incrementally, so the store stays append-only and needs
no side file. If no report exists, the pass starts from
`max(0, head - max_scan)`.

## Phase 0: Scheduler and trigger

- `DreamConfig` (library): `enabled`, `idle_after_secs` (default 900),
  `min_interval_secs` (default 3600), `max_scan` (default 500),
  `max_writes_per_pass` (default 20), `chains` (allowlist; empty means the
  default chain only), `dream_weight`, `llm: Option<LlmExtractionConfig>`,
  `recombination_budget` (default 3), and `seed: Option<u64>` for
  deterministic tests.
- Environment variables: `MENTISDB_DREAM_ENABLED`, `MENTISDB_DREAM_IDLE_SECS`,
  `MENTISDB_DREAM_INTERVAL_SECS`, `MENTISDB_DREAM_MAX_WRITES`,
  `MENTISDB_DREAM_WEIGHT`, `MENTISDB_DREAM_LLM` (reuses the existing LLM env
  configuration).
- A chain is **idle** when there has been no append by any agent other than
  `mentis-dreamer` for `idle_after_secs`, and the last pass was at least
  `min_interval_secs` ago.
- Server: one `tokio::spawn` loop with a coarse interval, alongside the
  existing background tasks. It has a cancellation token for clean shutdown,
  holds a per-chain lock so passes never overlap, and never blocks request
  handling.
- Manual trigger, which ignores idleness:
  - MCP `mentisdb_dream`, with `{ chain_key, dry_run, phases }`.
  - REST `POST /v1/dream`.
  - CLI `mentisdb dream [--chain] [--dry-run] [--phase consolidate,decay,recombine]`.
- `dry_run` returns the planned writes without appending anything. This is the
  main debugging tool.
- Webhook event `dream.completed` carries the report payload. A downstream
  consumer, such as a ComfyUI workflow, can render imagery from it. Image
  generation stays **outside** the crate.

## Phase 1: No-LLM consolidation and decay

### Salience sampler (`src/dream/salience.rs`)

The sampler scores thoughts in the scan window:

```
salience = importance
         * recency(age, half_life(type))
         * (1 + ln(1 + in_degree))
         * type_boost(type)
```

- `in_degree` counts refs and relations pointing at the thought, computed from
  the existing graph index. Phase 1 does **not** add read or access tracking,
  because that would mean writes on every read.
- `type_boost` is > 1 for `Surprise`, `Mistake`, `Correction`,
  `AssumptionInvalidated`, `Decision`, `PreferenceUpdate` and `UserTrait`.
  These are the "emotionally salient" analogs.
- The sampler skips `Dream`-role thoughts and invalidated thoughts. Dreams
  never consolidate other dreams in phase 1, which prevents feedback runaway.
- It is deterministic when a seed is given.

### Consolidation (extractive)

- Reuse `build_summary_candidates`, which already skips windows covered by
  `Summarizes`. Rank candidate windows by the summed salience of their members.
- For the top windows, within budget, append a `Summary`-type, `Dream`-role
  thought. Its content is an extractive digest: the highest-salience member
  statements, deduplicated and length-capped. Its tags and concepts are the
  union of the members'. Its importance is the maximum of the members'. It gets
  one `Summarizes` relation per member.

### Decay (query-time only)

- `effective_importance(thought, now)`: importance × exponential decay using
  the per-type half-life, with a floor so nothing reaches zero.
- A thought's age is refreshed by the timestamp of its newest inbound reference
  or summary, so being consolidated keeps a memory alive (the replay effect).
- The half-life table is a `pub const` with rustdoc and can be overridden in
  config:
  - Long: `Constraint`, `Decision`, `UserTrait`, `PreferenceUpdate`,
    `LessonLearned` (≈ 365 days).
  - Medium: `Finding`, `FactLearned`, `Insight` (≈ 90 days).
  - Short: `WorkingMemory` role, `Checkpoint`, `StateSnapshot` (≈ 7 days).
- Wire this into `search/ranked.rs` as an opt-in factor
  (`use_decay: bool`, default false in this release). Search-eval benchmarks
  must not regress when it is off.

### Dedup suggestions

- Near-duplicate pairs are those with vector cosine ≥ 0.95 **and** high
  lexical overlap, from the same agent scope and not already linked.
- For each pair, append one `Dream`-role `Finding` tagged `dream:suggestion`,
  with `RelatedTo` both. The content proposes which should supersede which
  (newer and higher importance wins).
- Nothing is superseded automatically.

## Phase 2: LLM-assisted (opt-in)

This phase runs only when `DreamConfig.llm` is set. It reuses the `llm.rs`
client, retries and error types, and adds prompts in `src/dream/prompts.rs`.

1. **Abstractive consolidation.** This upgrades phase 1 digests into true gist
   summaries. It uses the same selection, the same relations and a lower
   confidence cap.
2. **Recombination** (the REM analog):
   - Pick pairs of clusters that are both high-salience and **distant**
     (centroid cosine below a threshold). Clusters come from concepts or tags,
     or from vector k-means if that is cheap.
   - Prompt: "Here are two unrelated sets of memories. Propose at most N
     non-obvious, testable connections. Return JSON. Return an empty list if
     nothing is real."
   - Allowed output types are `Hypothesis`, `Idea`, `Wonder` and `Question`.
     Each output is a `Dream`-role thought with confidence ≤ 0.4 and
     `DerivedFrom` to the specific source thoughts it cites. Citations must be
     a subset of the ids that were shown to the model; otherwise the output is
     rejected.
   - A novelty filter drops any output with cosine ≥ 0.9 to an existing
     thought, and so drops restated facts.
   - The number of LLM calls per pass is capped by `recombination_budget`.
3. **Contradiction check.** For high-similarity pairs that phase 1 did not
   flag as duplicates, the LLM judges whether they contradict each other. If
   yes, it appends a `Dream`-role `Surprise` suggestion with `RelatedTo` both.
   It never appends `Contradicts` directly; that is left for promotion.
4. **Validation.** Enforce the type whitelist, cap content length, clamp
   importance and confidence, strip non-JSON output, and drop unknown ids. Log
   and count rejections in the pass report.

## Phase 3 (optional, after review): Dashboard "Dreams" tab

This tab lists dream thoughts grouped by pass, shows their sources inline, and
has Promote, Edit-and-promote, and Dismiss buttons that call the helpers above.
It also charts pass reports over time.

## Module layout

```
src/dream/mod.rs           DreamConfig, DreamReport, run_dream_pass(), pub API
src/dream/salience.rs      sampler, half-life table, effective_importance
src/dream/consolidate.rs   phase 1 extractive + phase 2 abstractive
src/dream/suggest.rs       dedup + contradiction suggestions
src/dream/recombine.rs     phase 2
src/dream/prompts.rs       phase 2 prompts
src/dream/scheduler.rs     idle detection + server loop (server feature)
tests/dream_tests.rs
```

## Test plan

- The role round-trips through serde, the parsers and the CLI. Old chains load
  unchanged.
- Dreams are excluded by default from `recent_context`, down-weighted in
  `ranked_search`, and included when `include_dreams` is set.
- The watermark works: a second pass with no new appends writes only a report.
  After K new appends, it scans only those.
- Dream thoughts never consolidate other dream thoughts.
- `dry_run` appends nothing.
- Budgets are respected (`max_writes_per_pass`, `recombination_budget`).
- Decay is off by default, and search-eval results are identical to `master`.
  With decay on, the ranking order changes as expected on a synthetic fixture.
- Dedup suggestions are created and nothing is superseded.
- Phase 2 uses a mock OpenAI-compatible server (same approach as
  `llm_extracted_memories_tests.rs`). Tests cover:
  - rejection of invalid JSON
  - rejection of citations outside the shown set
  - the novelty filter
  - confidence clamping
- The scheduler skips chains that are not idle, never overlaps passes, and
  shuts down cleanly.
- Chain integrity verification passes after dream passes. Signed chains stay
  valid.
- `cargo test`, `cargo clippy -- -D warnings` and the existing benches
  compile.

## Open questions for Track

1. Should dream thoughts go in the same chain or a sibling chain
   (`<chain>.dreams`) with `BranchesFrom`? This design uses the same chain with
   role filtering, which is simpler. A sibling chain gives harder isolation.
2. Should an agent see dreams in `recent_context` when it explicitly asks
   ("what have you been thinking about?")? That is currently opt-in only.
3. Which LLM should the dreamer use, the local GPU box or a hosted API? The
   answer sets the budget defaults.
4. Should phase 1 add access-count tracking (a sidecar, not chain writes) to
   improve salience scoring?
