# Kernel Harness — Project Context

An agentic loop that writes, tests, profiles, and iteratively optimizes GPU kernels for hackathons (primarily GPU Mode / gpumode.com). The agent does the kernel writing; the human sets stopping conditions and monitors progress.

## Glossary

- **Harness** — this project; the orchestration layer around the agent loop
- **Problem** — a leaderboard challenge fetched from `gpu-mode/reference-kernels` (has `reference.py`, `task.yml`, `task.py`)
- **Kernel** — a single Python submission file (may contain inline CUDA via `load_inline`, Triton, CUTLASS, etc.)
- **Iteration** — one full cycle: generate kernel → test → profile → reflect → rewrite
- **Library** — persistent cross-hackathon knowledge base of techniques, winning kernels, and failed approaches
- **Hardware profile** — auto-detected configuration that controls whether local profiling is available

## Entry Point

```bash
python harness.py run --leaderboard 543 --stop-on "2h|20iter|2x_reference"
```

Problem spec is fetched automatically from the GitHub API on `gpu-mode/reference-kernels` using the leaderboard ID. No manual problem description required.

## Tech Stack & Auth/Billing

Built on the **Claude Agent SDK** (`claude-agent-sdk`, Python), NOT the raw `anthropic` SDK. The Agent SDK drives the `claude` CLI under the hood, which gives us:
- Per-subagent model routing (`ClaudeAgentOptions.model`), `effort`, adaptive `thinking`, tool execution, custom tools (`@tool` + `create_sdk_mcp_server`), and a built-in `max_budget_usd` per-call cap.
- **Auth/billing via the CLI**: a Pro/Max subscription login draws from the plan's monthly **Agent SDK credit pool** (user is on **Max 5× = $100/mo**, separate from interactive Claude usage); an `ANTHROPIC_API_KEY` injected into the SDK env bills pay-per-token.

**Why this choice:** lets the user's subscription credit offset harness cost instead of pure pay-per-token, while preserving full per-subagent model routing. (Decided 2026-06-14. The Agent SDK credit-pool feature goes live 2026-06-15.)

**Billing modes** (`config.BillingConfig` / `AuthMode`):
- `subscription` — credit pool only; halt when exhausted.
- `api_key` — pure pay-as-you-go.
- `subscription_then_api_key` *(default)* — start on credit pool; on credit-exhaustion error, fall back to the configured API key for the rest of the run and flag it loudly in the TUI. (User explicitly wanted this graceful fallback.)

Orchestration shape is **hybrid**: Python code owns the deterministic loop (iterations, stopping conditions, parallel candidates); each subagent is a separate `query()` call with its own `model`/`effort`/tools. Runtime dep: the `claude` CLI must be installed (verified present: 2.1.177).

Open wiring detail (confirm at integration time, post-2026-06-15): exact mechanism that routes Agent SDK usage to the plan's credit pool vs. an API key, and the precise error/`RateLimitEvent` signal for credit exhaustion that triggers the fallback.

## Agent Architecture

Specialized subagents orchestrated by a top-level loop agent (each subagent = one Agent SDK `query()` with its own model):

| Subagent | Model | Role |
|---|---|---|
| Orchestrator | Sonnet | Loop control, evaluates stopping conditions each iteration |
| Problem understander | Haiku | Parses task.yml and reference.py into structured problem spec |
| Library retrieval | Sonnet | Semantic search over library for relevant techniques |
| Kernel writer | Opus | Generates/rewrites the Python+CUDA submission file |
| Profiler interpreter | Opus | Reads ncu/nsys output, identifies bottlenecks, produces actionable findings |
| Reflection/strategy | Opus | Decides what to try next based on full iteration history |
| Library updater | Haiku | Writes new techniques, results, and failed approaches back to library |

## Kernel Generation Strategy

- **Iteration 1**: Generate multiple parallel candidates (Triton, raw CUDA, CUTLASS) — explore the search space
- **Iteration 2+**: Single focused rewrite based on the best candidate + profiler findings

## Stopping Conditions (combinable)

- `manual` — user intervenes via TUI
- `<Nx_reference>` — stop when kernel beats N× the reference implementation speed
- `<Niter>` — stop after N iteration cycles
- `<Nh>` / `<Nm>` — stop after N hours/minutes (wall clock)
- Combinations short-circuit on the first condition met

## Profiling Strategy

- **Local ncu/nsys output** drives rewrite decisions (rich signal, free, fast)
- **Leaderboard score** (from `popcorn submit --mode benchmark`) is ground truth for stopping decisions
- The agent never optimizes purely for leaderboard score without understanding *why* via ncu

## Hardware Profiles

Auto-detected via `nvidia-smi` at startup; manually overridable via `--hardware` flag.

| Profile | Local test | Local profiling | Context |
|---|---|---|---|
| `blackwell` | Yes | ncu + Blackwell spec sheet | GB10 / DGX Spark |
| `nvidia` | Yes | ncu + generic CUDA context | Any NVIDIA non-Blackwell |
| `leaderboard-only` | No | None — profiler interpreter skipped | CPU or non-NVIDIA |

A `hardware/blackwell-spec.md` document is loaded as permanent context for kernel writer and profiler interpreter when on the `blackwell` profile.

## Knowledge Library

Persists across hackathons. Stored as:
- **Raw files** (`library/`) — human-readable, version-controlled, editable
- **Vector DB** (ChromaDB) — semantic retrieval for the library retrieval subagent

Four categories:
1. **Techniques** — e.g. "shared memory tiling for matmul," "vectorized loads with float4"
2. **Winning kernels** — annotated with problem, GPU, score achieved
3. **Failed approaches** — what was tried, why it didn't work
4. **Hardware-specific notes** — observations tied to a specific GPU architecture

## Submission

Single Python file submitted via `popcorn-cli`:
```bash
popcorn submit --leaderboard <name> --gpu <gpu> --mode test|benchmark|leaderboard solution.py
```
Supported libraries: raw CUDA (via `load_inline`), Triton, PyTorch, ThunderKittens, cuDNN, CUTLASS, and others depending on hackathon.

## Interaction Model

- **Default**: fire-and-forget autonomous loop
- **TUI**: attachable at any time to inspect iteration history, current kernel, profiler output; can intervene, change strategy, or force-submit

## Repo Layout

```
/
├── CONTEXT.md                       ← this file (authoritative design)
├── pyproject.toml                   ← package metadata + deps
├── src/kernel_harness/
│   ├── __init__.py
│   ├── config.py                    ← HardwareProfile + detection, StoppingConditions, RunConfig, model+effort routing, AuthMode/BillingConfig
│   ├── problem.py                   ← ProblemFetcher: leaderboard name → spec_files + stage_files + tests/benchmarks (from task.yml) via GitHub API
│   ├── agent.py                     ← AgentRunner (cloud/Claude): one query() per subagent; model/effort/thinking routing + credit→api-key fallback
│   ├── backends.py                  ← LocalRunner (OpenAI-compatible Ollama/vLLM) + BackendRouter: per-subagent cloud/local dispatch with cloud fallback
│   ├── evalproto.py                 ← GPU MODE local-eval protocol: stage workspace, FD-plumbed run_eval, parse key:value output (test/benchmark)
│   ├── tools.py                     ← MCP tool layer (RunContext + build_tool_server): save_submission, run_local, profile_kernel, popcorn_submit
│   ├── subagents.py                 ← the six subagents: prompt builders + typed results parsed from JSON; understand_problem/retrieve_knowledge/write_kernel/interpret_profile/reflect/update_library
│   ├── orchestrator.py              ← the loop: baseline → understand → retrieve → parallel candidates → serial ground-truth eval → profile → reflect → rewrite; StoppingConditions; per-candidate RunContext isolation; persists lessons to library
│   ├── library.py                   ← cross-hackathon knowledge store: raw JSON files + retrieval index (ChromaDB or pure-Python lexical fallback); persist_entries / candidates_for
│   ├── seeds.py                     ← curated transferable Blackwell/Triton lessons (paraphrased from Dogacel/auto-gpu-kernel, Apache-2.0) + seed_library()
│   ├── ledger.py                    ← persistent on-disk experiment record: runs/<id>/summary.md + per-candidate result.md; reflect/update_library read render()
│   ├── cli.py                       ← entry point: `kernel-harness run --leaderboard ... --stop-on ...` and `list`; build_parser/build_run_config (pure) + rich event printer; `--tui` launches the dashboard
│   └── tui.py                       ← Textual dashboard: candidate table + event log + force-stop; consumes Orchestrator.on_event, calls request_stop()
├── tests/
│   ├── test_foundation.py           ← config + problem-fetcher smoke tests
│   ├── test_agent.py                ← agent wrapper: exhaustion helpers, option building, billing fallback (faked query, offline)
│   ├── test_evalproto.py            ← eval protocol + tools: parsers, command builders, end-to-end run_eval vs stub eval.py, MCP server assembly
│   ├── test_subagents.py            ← six subagents via a fake runner: JSON recovery, typed results, prompt/tool routing
│   ├── test_orchestrator.py         ← full-loop integration (faked runner + stubbed GPU boundaries): parallel candidates, best-selection, stop paths
│   ├── test_cli.py                  ← CLI arg→config mapping + event-printer robustness
│   ├── test_library.py              ← library: lexical ranking, file roundtrip, persist_entries, candidates_for, idempotency, orchestrator-persists-lessons
│   ├── test_tui.py                  ← TUI: event→row/log mappers + run_test app smoke (table fills, force-stop, final report)
│   └── test_backends.py             ← local backend: config routing, OpenAI-compatible call shape (mock transport), router dispatch + cloud fallback
└── docs/agents/                     ← issue-tracker / triage-labels / domain config (from setup skill)
```

Run tests: `for t in test_foundation test_agent test_evalproto test_subagents test_orchestrator test_cli test_library test_tui test_backends test_seeds test_ledger test_research test_seed_kernel; do PYTHONPATH=src .venv/bin/python -m tests.$t; done` (78 checks, all offline except one live GitHub fetch)

Try it: `PYTHONPATH=src .venv/bin/python -m kernel_harness.cli list` (works now); `... cli run --leaderboard vectoradd_v2 --stop-on "20iter|2x_reference" [--tui]` (needs GB10 + Agent SDK auth).

## Build Status (as of 2026-06-14)

**Done & tested:**
- Project scaffolding (`pyproject.toml`, `src/` layout, `.gitignore`)
- `config.py` — hardware-profile auto-detection (verified: detects `blackwell` on the GB10), `--stop-on` parser (`2h|20iter|2x_reference` → combined OR-semantics conditions), `RunConfig`, default per-subagent model routing, and **`AuthMode`/`BillingConfig`** (subscription / api_key / subscription_then_api_key with graceful fallback — tested)
- `problem.py` — `ProblemFetcher` resolves a leaderboard *name* (not numeric id) against `gpu-mode/reference-kernels`, downloads `task.yml`/`task.py`/`reference.py` (verified live against the public GitHub API)
- Architecture decided: **Claude Agent SDK** (`claude-agent-sdk` 0.2.101 installed; `claude` CLI 2.1.177 present). SDK API introspected — `query()` + `ClaudeAgentOptions(model, effort, thinking, agents, max_budget_usd, tools, env)` give us everything the design needs.
- `agent.py` — **`AgentRunner`**: `run_subagent(name, prompt, ...)` issues one `query()` with routed model + effort + adaptive thinking + scoped tools, applies `BillingConfig` auth, streams text/events via callbacks, and on credit exhaustion (detected via `AssistantMessage.error in {billing_error, rate_limit}`, rejected `RateLimitEvent`, or 402/429 `ResultMessage`) falls back subscription→api-key once and emits a notice. Effort routing added to config. Tested offline with a faked `query` (`tests/test_agent.py`).
- `evalproto.py` + `tools.py` — **local tool layer**. `evalproto` reproduces GPU MODE's eval harness exactly: stages a workspace from `Problem.stage_files`, generates the `key: val; ...` test-spec file from `tests`/`benchmarks`, runs `python eval.py <mode> <testfile>` with `POPCORN_FD` plumbing, and parses `check`/`test.*`/`benchmark.*` output into `EvalResult` (geomean ns score). `tools.py` wraps four MCP tools (`save_submission`, `run_local`, `profile_kernel`, `popcorn_submit`) bound to a `RunContext`, hardware-gated, returned by `build_tool_server(ctx) -> (mcp_servers, allowed_tools)` ready for `AgentRunner`. Tested offline incl. end-to-end `run_eval` against a stub eval.py (`tests/test_evalproto.py`).
  - ⚠️ **On-device validation pending:** the *real* `run_local` (needs torch+CUDA), `profile_kernel` (`ncu --set basic ...`), and `popcorn_submit` (popcorn-cli flags/auth) can only be fully exercised on the GB10. Protocol/plumbing/parsing are validated; the exact ncu set + popcorn flags should be confirmed on the first real run.

- `subagents.py` — the **seven subagents** (six core + `workload_inspector`) as async functions over `AgentRunner`, each with a focused system prompt + tool scope + typed result parsed from a fenced ```json block (`parse_json_block`, offline-robust). `understand_problem` (Haiku, no tools), `retrieve_knowledge` (Sonnet; short-circuits to empty when no library entries — takes candidate note strings, so it doesn't depend on the library being built yet), `write_kernel` (Opus; kernel_tools + Write/Read/Edit/Bash; takes `approach` for parallel candidates + per-call `mcp_servers`/`allowed_tools` so candidates get isolated workspaces; accepts knowledge/prior_summary/profiler findings), `interpret_profile` (Opus; profile_kernel/run_local), `reflect` (Opus; decides iterate/submit/stop), `update_library` (Haiku; emits structured entries). Added per-call `mcp_servers` override to `AgentRunner`. Tested offline with a fake runner (`tests/test_subagents.py`, 8 checks).

- `orchestrator.py` — **`Orchestrator.run()`** ties it together: fetch problem → `understand_problem` → establish a **reference baseline** (benchmark `from reference import ref_kernel as custom_kernel` locally; speedup = reference_ns/candidate_ns) → `retrieve_knowledge` → iteration 1 writes `config.candidate_approaches` candidates **in parallel** (each its own workspace + tool server via `build_tool_server`) → **serial** ground-truth eval through `evalproto` (trusts measured numbers, not agent self-report) → iter 2+ `interpret_profile` best → `reflect` (iterate/submit/stop) → focused `write_kernel` rewrite (workspace seeded from current best). Evaluates `StoppingConditions` (time/iter/speedup) + external stop event (`request_stop()` for the TUI) + safety cap each iteration. `auto_submit` off by default (ranked submission is outward-facing — reports best, lets user/TUI fire it). Emits events via `on_event(name, payload)` for the TUI. Added `candidate_approaches` + `auto_submit` to RunConfig. Pure helpers (`compute_speedup`, `select_best`, `render_history`) + full-loop integration tested offline (`tests/test_orchestrator.py`, 6 checks).
  - ⚠️ **First real end-to-end run happens here** — needs the GB10 (torch+CUDA for `run_eval`, ncu, popcorn auth) and live Agent SDK billing. The control flow + decision logic are tested; the live integration (real `ref_kernel` import, popcorn output, credit-pool wiring) is validated on first run.

- `cli.py` — entry point. `kernel-harness run --leaderboard <name> --stop-on <spec>` (+ `--gpu/--hardware/--auth-mode/--max-usd-per-query/--approaches/--auto-submit/--library-dir/--no-library`) and `kernel-harness list`. Pure `build_parser`/`build_run_config` (tested); resolves hardware via `detect_hardware_profile`, API key from `ANTHROPIC_API_KEY`, default GPU from the problem; rich event printer streams progress. `list` verified live. Tested (`tests/test_cli.py`).
- `library.py` — cross-hackathon knowledge store. Raw JSON files under `<root>/<kind>/` (technique/failed_approach/winning_kernel/hardware_note) + a retrieval index that is **ChromaDB when importable, else a pure-Python lexical scorer** (so a fresh machine with no embedding model still works). `persist_entries(LibraryEntries)` ingests `update_library` output (content-hash IDs → idempotent); `candidates_for(brief)` feeds `library_retrieval`. Wired into `Orchestrator(library=...)`: supplies retrieval candidates and receives distilled lessons at run end (best-effort, never fails a run). Tested (`tests/test_library.py`).

- `tui.py` — Textual dashboard (`--tui`). Candidate `DataTable` + event `RichLog` + status bar; `s` force-stops (`request_stop`), `q` quits. Decoupled from the orchestrator (duck-typed `set_on_event`/`request_stop`/`run`). Pure `candidate_row`/`format_log_line`/`format_report` + a `run_test` app smoke test (`tests/test_tui.py`).

**All originally-designed components are now built.** The harness runs headless (`kernel-harness run ...`) or with the dashboard (`--tui`).

**Post-design additions (informed by Dogacel/auto-gpu-kernel — the winning MLSys 2026 agent-only kernel-gen entry, Apache-2.0):**
- `seeds.py` + `kernel-harness seed-library` — curated, *transferable* Blackwell/Triton lessons (split-K for SM-starved grids, scalar-`if` kills num_stages prefetch, num_warps=8 for H=16 MLA, `.cg`/`evict_first`/`evict_last` L2 hints, online-softmax NaN guard, `.item()` in launcher is catastrophic, Gluon `dot_fma` has no tensor cores, etc.) paraphrased + attributed (`NOTICE`). `library/` is now gitignored (runtime state); the generator is committed.
- `workload_inspector` subagent (7th) — analyzes the problem's *input distribution* for regime-specific shortcuts (the source's biggest win lever). Runs once after `understand_problem`; its `WorkloadProfile` (regimes/structure/shortcuts) feeds every `write_kernel` prompt. Routed Sonnet/medium/cloud; wired in `Orchestrator` (`self._workload`, `report.workload`).
- `ledger.py` — **persistent on-disk experiment ledger** (built; the chosen "next" item). Each run writes `runs/<run_id>/summary.md` (header: problem/brief/GPU/reference baseline/workload structure, then one table row per evaluated candidate + per-iteration profiler/decision notes) and per-candidate `result.md` next to its `submission.py` snapshot. `Orchestrator` creates it after `inspect_workload`, records each candidate (with a `**best**` marker) + iteration notes, and `reflect`/`update_library` now read `ledger.render()` from disk instead of in-memory `render_history`. Value: resumability/inspectability, bounded prompt size on long runs, and the substrate a future research agent reads via file tools. Best-effort writes (never fail a run). Tested (`tests/test_ledger.py`, incl. orchestrator integration).
- **Smarter reflect cadence + research agent** (built). `reflect` stays the cheap per-iteration decider (now "judge by ceiling not iteration-0"). A new **`research`** subagent (7→8th; Opus/high/cloud) is fired ONLY when `detect_stuck(iterations)` triggers — **plateau** (best speedup improved <5% over the last 4 iterations) or **correctness wall** (no passing candidate in 3 iterations). It's clean-context: from the on-disk ledger + current best kernel + library lessons it runs a pathology checklist (repetition loop / local minimum / correctness wall / wrong bottleneck / missing fundamental / over-engineering / overlooked shortcut) and emits a `ResearchPlan` (diagnosis, strategy pivot/refactor/targeted, prioritized actions, **do_not_try** dead-ends). The plan feeds `reflect`'s decision AND the next `write_kernel` rewrite (and `do_not_try` warns it off cycles); recorded in the ledger + surfaced as a `research` event in CLI/TUI. `detect_stuck` is a pure, tested helper. Tested (`tests/test_research.py`, incl. plateau→research→rewrite integration).
**DONE (2026-06-15): `--seed-kernel` warm-start.** Implemented exactly as designed below. `RunConfig.seed_kernel` + `--seed-kernel <path>` CLI flag; `Orchestrator._prepare_seed` reads + benchmarks the seed before iteration 1 — if it PASSES it's injected into the iteration-1 pool as a competing `CandidateRecord(id="seed")` (not re-evaluated in the loop), if it FAILS its code becomes `self._seed_reference` threaded into the cold writers' prompts as a structural reference (`build_write_prompt`/`write_kernel` `seed_reference=`). Emits a `seed` event (CLI/TUI). Anti-anchor proven in tests: a passing seed loses to a faster cold candidate; a failing seed flows to writers and never enters the pool. `tests/test_seed_kernel.py` (4 checks). 78 tests / 13 suites green. STILL DEFERRED: `--seed-notes`/`library add` methodology import.

**Original approved design (for reference): `--seed-kernel` warm-start.**
Let the user supply a starting `submission.py` (authored interactively in Claude Code / Nsight Copilot / by hand) so the harness optimizes *from* it instead of cold-generating — saves tokens (fewer/cheaper iteration-1 candidates; interactive authoring draws on the user's *direct* subscription, a different quota than the harness credit pool) and gives control. User's key requirement (resolved): the seed must be a **competing starting candidate + structural template, NOT an anchor** (their fear: "start bad → end bad"). Approved design — `--seed-kernel <path>`:
1. **Always measured, never trusted** — benchmark the seed (correctness + timing vs the reference baseline). If it FAILS correctness, don't promote it; hand it to the kernel writers as a *structural reference* ("here's a starting point — fix/build on it"). If it PASSES, add it to the iteration-1 pool as a candidate (mark it `seed`).
2. **Competes, doesn't dictate** — the seed enters iteration 1 *alongside* freshly generated candidates (so a bad seed gets beaten by a cold one, not polished). Token savings come from the user lowering `--approaches` (e.g. seed + 1 fresh) + faster convergence, NOT from skipping cold generation (that's the risky "refine-only" mode the user explicitly rejected).
3. **Loop can abandon the seed's lineage** — the already-built `detect_stuck`→`research`→pivot machinery escapes a plateaued bad seed automatically ("judge by ceiling not current").
Implementation sketch: `RunConfig.seed_kernel: Path|None` + `--seed-kernel` CLI flag; orchestrator stages+evaluates the seed before iteration 1, injects it into the iteration-1 pool as a `CandidateRecord` (id `seed`), or threads its code into the kernel-writer prompt as a structural reference when it fails correctness; ledger marks provenance (`seed: user-provided`); if a seeded kernel wins, the library entry notes human origin. DEFERRED (user said decide after seed): a `--seed-notes`/`library add` methodology-import command (let the user import their own Nsight/Claude-Code techniques into the library). 

- Still TODO (lowest priority): an `ab_benchmark` paired-rerun tool + one-change-per-iteration discipline — marginal on the single GB10 (low run-to-run noise); revisit only if the first real run shows it's needed.

- `backends.py` — **hybrid cloud+local compute**. `LocalRunner` calls an OpenAI-compatible endpoint (Ollama/vLLM via httpx — `/v1/chat/completions`, tool-free) on the GB10; `BackendRouter` implements the same `run_subagent` interface so it's a drop-in "runner" (zero change to subagents/orchestrator) and dispatches per subagent via `RunConfig.backend_for`, **falling back to cloud per-call** (and disabling local for the rest of the run) if the endpoint errors — so "local by default" never breaks a run. Defaults (`DEFAULT_BACKEND_ROUTING`): problem_understander / library_retrieval / library_updater → **local**; everything else (incl. tool-using kernel_writer / profiler_interpreter) → **cloud**. Config gained `backend_routing` + `LocalConfig` (base_url/model/timeout) + `backend_for`/`uses_local`; `SubagentResult` gained `backend`. CLI flags: `--no-local`, `--local-base-url`, `--local-model`, `--local-subagents` (tool-using roles forced cloud even if listed). Tested with a mock httpx transport (`tests/test_backends.py`). **Why:** the free local models absorb mechanical work so the subscription credit stretches to the kernel-writing that matters.

**Remaining work:**
- ⚠️ **On-device validation** (unchanged): first real GB10 run validates live `run_eval`/ncu/popcorn + the Agent SDK credit-pool wiring (the June-15 billing feature) + the local Ollama/vLLM endpoint (a model must be pulled, e.g. `ollama pull qwen2.5-coder:7b`).

**Open design notes:**
- Numeric leaderboard id (e.g. 543) → name (e.g. `vectoradd_v2`) resolution is NOT implemented; the harness keys on the name (matches `popcorn submit --leaderboard <name>`). Numeric-id lookup would need the gpumode.com web API (JS-rendered; not in the reference repo).
- `popcorn-cli` submission modes: `test` (correctness), `benchmark` (perf, unranked), `leaderboard` (ranked), `profile` (ncu). Single Python file per submission.
