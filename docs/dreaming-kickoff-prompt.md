# Kickoff prompt for Claude Code

Open Claude Code in `E:\repos\mentisdb` on branch `feature/dreaming` and paste:

---

You're implementing "dreaming" for MentisDB: offline, idle-time consolidation
of memory. The full design is in `docs/dreaming-design.md`. Read it, then
`AGENTS.md`, before writing any code. The non-negotiables in the design are
hard constraints:

- storage stays append-only
- the no-LLM core stays intact
- every dream write carries provenance
- a dream pass suggests changes and never supersedes or invalidates anything
- everything is off by default
- new public API gets rustdoc, and new behavior gets tests

Work in phases, and stop for my review after each one:

1. **Phase 0 + role plumbing.** Add `ThoughtRole::Dream` everywhere roles are
   parsed or displayed. Add `include_dreams` to the query types, following the
   `include_invalidated` pattern. Add `DreamConfig`, the pass report and
   watermark, the `mentisdb_dream` MCP tool, the REST endpoint, the CLI
   command with `--dry-run`, the idle scheduler loop, and the
   `dream.completed` webhook event.
2. **Phase 1.** Add the salience sampler, extractive consolidation built on
   `build_summary_candidates`, query-time decay wired into ranked search
   behind `use_decay` (default off), and dedup suggestions.
3. **Phase 2.** Add LLM abstractive consolidation, recombination with
   citation and novelty validation, and contradiction suggestions. All of it
   goes through the existing `llm.rs` client, and tests use a mock server.

Before starting each phase, use plan mode, explore the code the phase touches
(`src/lib.rs` is large, so use targeted searches), and show me the plan. After
each phase:

- run `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and
  `cargo test`
- confirm that the search-eval tests don't regress with decay off
- update `README.md`, `MENTISDB_SKILL.md` and `changelog.txt`
- make one focused commit on `feature/dreaming`

Note: the working tree shows most files as modified because of CRLF line
endings only (`git diff --ignore-cr-at-eol` is empty). Don't commit that
noise. Stage only the files you change, and check with
`git diff --cached --stat` before each commit.

If you hit a design question the doc leaves open (see "Open questions"), ask
me instead of guessing.
