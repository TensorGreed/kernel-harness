# kernel-harness

An agentic loop that writes, tests, profiles, and iteratively optimizes GPU
kernels for [GPU MODE](https://www.gpumode.com) hackathons. You point it at a
leaderboard problem and set stopping conditions; specialized Claude subagents
(orchestrated by a deterministic loop) write candidate kernels, benchmark them
locally, read the profiler, and rewrite — accumulating a cross-hackathon
knowledge library as they go.

The design and internals live in [CONTEXT.md](./CONTEXT.md).

## How it works (30 seconds)

```
fetch problem ─▶ understand ─▶ baseline (reference) ─▶ retrieve prior techniques
      │
      ▼
  iteration 1: write N candidates in parallel (triton / cuda-inline / pytorch)
      │          ▼ benchmark each locally (ground truth)
  iteration 2+: profile best ─▶ decide (iterate/submit/stop) ─▶ focused rewrite
      │
      ▼
  stop on: time | iterations | Nx-reference | reflection | you (TUI)
      └▶ report best kernel · persist lessons to the library
```

Subagents are routed per role: **Opus** writes/optimizes kernels and reads the
profiler, **Sonnet/Haiku** handle lighter work, and the mechanical, tool-free
roles can run **free on local Ollama/vLLM** models on the GPU box.

## Prerequisites

- **Python 3.10+**
- **[Claude Code CLI](https://code.claude.com)** (`claude`) — the Agent SDK
  drives it. Log in with your Claude Pro/Max subscription (`claude` → `/login`)
  or set `ANTHROPIC_API_KEY`.
- **For local testing/profiling** (the GB10 / any NVIDIA GPU): a working
  PyTorch + CUDA install, and `ncu` (Nsight Compute) for profiling. Without a
  GPU the harness still runs in `leaderboard-only` mode (it submits via popcorn
  instead of testing locally).
- **[popcorn-cli](https://github.com/gpu-mode/popcorn-cli)** (`popcorn`) to
  submit to the leaderboard — install it and authenticate (`popcorn register
  discord` or `github`).
- **Optional — local models:** [Ollama](https://ollama.com) (or a vLLM server)
  to offload the mechanical subagents. e.g. `ollama pull qwen3:30b`.

## Install

```bash
git clone <this-repo> && cd kernel-harness
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

That installs the dependencies (Claude Agent SDK, Textual, rich, ChromaDB, …)
and the `kernel-harness` command.

> ChromaDB powers semantic library search. If it isn't available the library
> transparently falls back to a dependency-free lexical search — nothing breaks.

## Authentication & billing

The harness reads your auth from the `claude` CLI / environment and supports
three billing modes (`--auth-mode`):

| Mode | Behavior |
|------|----------|
| `subscription` | Use your plan's monthly Agent SDK credit pool; stop when exhausted. |
| `api_key` | Pay-per-token via `ANTHROPIC_API_KEY`. |
| `subscription_then_api_key` *(default)* | Start on subscription credit; on exhaustion, fall back to `ANTHROPIC_API_KEY` for the rest of the run (announced in the UI). |

To run purely on subscription credit, make sure `ANTHROPIC_API_KEY` is **unset**
(an ambient key makes the CLI bill per-token).

## First run

Seed the knowledge library with curated, transferable Blackwell/Triton lessons
(optional but recommended — makes the first run smarter):

```bash
kernel-harness seed-library
```

List the available leaderboard problems (canonical names):

```bash
kernel-harness list
```

Optimize one, stopping at whichever comes first — 10 iterations or 2× the
reference implementation's speed:

```bash
kernel-harness run --leaderboard vectoradd_v2 --stop-on "10iter|2x_reference"
```

Add the live dashboard (candidate table + event log; `s` to force-stop, `q` to
quit):

```bash
kernel-harness run --leaderboard vectoradd_v2 --stop-on "2h|20iter|2x_reference" --tui
```

The run writes each candidate's workspace under `runs/<timestamp>/` and prints
the best kernel's path. Ranked leaderboard submission is **off by default** —
the report shows the exact `popcorn submit` command, or pass `--auto-submit` to
fire it automatically.

### Warm-start from your own kernel (`--seed-kernel`)

Bring a starting `submission.py` you wrote interactively (in Claude Code, guided
by Nsight Copilot, or by hand) and let the harness optimize from it:

```bash
kernel-harness run -l vectoradd_v2 --stop-on "20iter|2x_reference" \
  --seed-kernel ./my_kernel.py --approaches triton
```

The seed is a **competing candidate, never an anchor**: the harness benchmarks it
itself; if it passes it enters iteration 1's pool (and can be beaten by a fresh
candidate), and if it fails it's handed to the kernel writers as a structural
reference to fix and build on. Either way the profile→research→pivot loop can
walk away from it — so a weak seed can't trap the run. Saves tokens (bring one
strong candidate, run fewer cold ones) and keeps you in control of the approach.

### Stopping conditions (`--stop-on`)

Combine with `|`; the loop stops on the first one met:

- `2h` / `30m` / `90s` — wall-clock budget
- `20iter` — iteration budget
- `2x_reference` (or `2x`) — speedup over the reference implementation
- `manual` — only you stop it (via the TUI)

## Local models (optional)

By default the tool-free subagents (`problem_understander`,
`library_retrieval`, `library_updater`) run on a local OpenAI-compatible
endpoint, and the rest stay on Claude. If the endpoint isn't reachable, each
call transparently falls back to cloud.

```bash
# default: Ollama at localhost:11434, model qwen3:30b
kernel-harness run -l vectoradd_v2 --stop-on 10iter

# point at vLLM, pick a model, or change which subagents go local
kernel-harness run -l vectoradd_v2 \
  --local-base-url http://localhost:8000/v1 \
  --local-model llama3.3 \
  --local-subagents problem_understander,library_updater

# all cloud
kernel-harness run -l vectoradd_v2 --no-local
```

The kernel writer and profiler always run on Claude (they need the tool loop).

## Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--gpu` | problem's first GPU | leaderboard target GPU |
| `--hardware` | auto-detect | `blackwell` / `nvidia` / `leaderboard-only` |
| `--approaches` | `triton,cuda-inline,pytorch` | parallel candidates on iteration 1 |
| `--auto-submit` | off | fire a ranked submission when finished |
| `--library-dir` | `library` | cross-hackathon knowledge store |
| `--max-usd-per-query` | none | hard per-subagent USD cap |

## The knowledge library

Lessons (techniques, winning kernels, failed approaches) are distilled at the
end of each run and stored under `library/` as editable JSON, indexed for
retrieval on future runs. It grows with every hackathon. Seed it with
`kernel-harness seed-library` (curated Blackwell/Triton lessons; see `NOTICE`
for attribution). Disable per-run with `--no-library`.

## Acknowledgements

The seeded optimization lessons are paraphrased from
[Dogacel/auto-gpu-kernel](https://github.com/Dogacel/auto-gpu-kernel) (Apache-2.0,
© Doğaç Eldenk) — the winning agent-only entry of the MLSys 2026 FlashInfer
kernel-generation contest. See `NOTICE`.

## Development

```bash
# run the test suite (offline except one live GitHub fetch)
for t in test_foundation test_agent test_evalproto test_subagents \
         test_orchestrator test_cli test_library test_tui test_backends; do
  PYTHONPATH=src python -m tests.$t
done
```

Without an editable install, run the CLI via `PYTHONPATH=src python -m
kernel_harness.cli ...`.
