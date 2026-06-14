"""Command-line entry point.

    kernel-harness run --leaderboard vectoradd_v2 --stop-on "2h|20iter|2x_reference"
    kernel-harness list

Builds a :class:`RunConfig` (auto-detecting the hardware profile and reading the
API key from the environment), then runs ``Orchestrator.run()`` and prints a
report. Progress streams to the console via the orchestrator's event hook until
the Textual TUI lands.

The arg→config mapping is factored into pure functions (``build_parser`` /
``build_run_config``) so it can be unit-tested without touching the network,
a GPU, or the model. See CONTEXT.md → "Entry Point".
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .config import (
    DEFAULT_BACKEND_ROUTING,
    AuthMode,
    BillingConfig,
    HardwareProfile,
    LocalConfig,
    RunConfig,
    StoppingConditionError,
    StoppingConditions,
    detect_hardware_profile,
)
from .problem import ProblemFetcher, ProblemNotFoundError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernel-harness",
        description="Agentic loop that writes, tests, and optimizes GPU kernels for GPU MODE hackathons.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="optimize a leaderboard kernel")
    run.add_argument(
        "--leaderboard", "-l", required=True,
        help="canonical leaderboard name, e.g. 'vectoradd_v2' (see `kernel-harness list`)",
    )
    run.add_argument(
        "--stop-on", default="10iter",
        help="combined stop conditions, e.g. '2h|20iter|2x_reference' (default: 10iter)",
    )
    run.add_argument(
        "--gpu", default=None,
        help="leaderboard target GPU (default: the problem's first listed GPU)",
    )
    run.add_argument(
        "--hardware", choices=[h.value for h in HardwareProfile], default=None,
        help="local hardware profile (default: auto-detect via nvidia-smi)",
    )
    run.add_argument(
        "--auth-mode", choices=[a.value for a in AuthMode],
        default=AuthMode.SUBSCRIPTION_THEN_API_KEY.value,
        help="billing source (default: subscription_then_api_key)",
    )
    run.add_argument(
        "--max-usd-per-query", type=float, default=None,
        help="hard per-subagent USD cap (passed to the SDK)",
    )
    run.add_argument(
        "--approaches", default="triton,cuda-inline,pytorch",
        help="comma-separated approaches to try in parallel on iteration 1",
    )
    run.add_argument(
        "--auto-submit", action="store_true",
        help="fire a ranked leaderboard submission when finished (default: off)",
    )
    run.add_argument("--runs-dir", default="runs", help="where to write per-run workspaces")
    run.add_argument(
        "--library-dir", default="library",
        help="cross-hackathon knowledge library directory (default: ./library)",
    )
    run.add_argument(
        "--no-library", action="store_true",
        help="disable the knowledge library for this run",
    )
    run.add_argument(
        "--tui", action="store_true",
        help="launch the live Textual dashboard instead of streaming to stdout",
    )
    # Local-model backend (Ollama / vLLM) for the mechanical, tool-free subagents.
    run.add_argument(
        "--no-local", action="store_true",
        help="run every subagent on cloud Claude (disable the local backend)",
    )
    run.add_argument(
        "--local-base-url", default="http://localhost:11434/v1",
        help="OpenAI-compatible local endpoint (default: Ollama at localhost:11434/v1)",
    )
    run.add_argument(
        "--local-model", default="qwen3:30b",
        help="default local model name for routed subagents",
    )
    run.add_argument(
        "--local-subagents", default=None,
        help="comma-separated subagents to run locally (overrides the default routing); "
             "tool-using roles (kernel_writer, profiler_interpreter) always stay on cloud",
    )

    sub.add_parser("list", help="list all known leaderboard problem names")

    seed = sub.add_parser("seed-library", help="populate the library with curated Blackwell/Triton lessons")
    seed.add_argument("--library-dir", default="library", help="library directory to seed")
    return p


def build_run_config(
    args: argparse.Namespace,
    *,
    gpu: str,
    hardware: HardwareProfile,
    api_key: str | None,
) -> RunConfig:
    """Map parsed args + resolved environment into a RunConfig (pure)."""
    billing = BillingConfig(
        mode=AuthMode(args.auth_mode),
        api_key=api_key,
        max_usd_per_query=args.max_usd_per_query,
    )
    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]

    # Local backend (Ollama/vLLM). Tool-using roles are never routed local.
    local = None
    backend_routing = dict(DEFAULT_BACKEND_ROUTING)
    if not getattr(args, "no_local", False):
        local = LocalConfig(base_url=args.local_base_url, default_model=args.local_model)
        if args.local_subagents is not None:
            wanted = {s.strip() for s in args.local_subagents.split(",") if s.strip()}
            tool_using = {"kernel_writer", "profiler_interpreter"}
            backend_routing = {
                name: ("local" if name in wanted and name not in tool_using else "cloud")
                for name in DEFAULT_BACKEND_ROUTING
            }

    return RunConfig(
        leaderboard=args.leaderboard,
        gpu=gpu,
        stop_on=StoppingConditions.parse(args.stop_on),
        hardware=hardware,
        billing=billing,
        candidate_approaches=approaches or ["pytorch"],
        auto_submit=args.auto_submit,
        backend_routing=backend_routing,
        local=local,
    )


# --------------------------------------------------------------------------- #
# Console reporting
# --------------------------------------------------------------------------- #
def _make_event_printer():
    """Return an ``on_event(name, payload)`` that prints readable progress."""
    try:
        from rich.console import Console

        console = Console()
        emit = console.print
    except ImportError:  # rich is a dep, but degrade gracefully
        import re

        _markup = re.compile(r"\[/?[a-z0-9 ]*\]")

        def emit(msg: str) -> None:  # type: ignore[misc]
            print(_markup.sub("", msg))

    def on_event(name: str, payload: dict) -> None:
        if name == "run_start":
            emit(f"[bold]▶ optimizing[/] {payload['problem']} on {payload['gpu']}")
        elif name == "brief":
            emit(f"  understood: {payload['summary']}")
        elif name == "baseline":
            ref = payload.get("reference_ns")
            emit(f"  reference baseline: {ref:.1f} ns" if ref else "  reference baseline: unavailable")
        elif name == "iteration_start":
            emit(f"[bold cyan]── iteration {payload['index']}[/]")
        elif name == "candidate_evaluated":
            spd = payload.get("speedup")
            spd_s = f"{spd:.2f}x" if spd else "n/a"
            status = "✓" if payload["passed"] else "✗"
            emit(f"  {status} {payload['id']} ({payload['approach']}): speedup={spd_s}")
        elif name == "candidate_failed":
            emit(f"  ✗ {payload['approach']} failed: {payload['error']}")
        elif name == "decision":
            emit(f"  strategy: [yellow]{payload['action']}[/] — {payload.get('focus','')}")
        elif name == "notice":
            emit(f"  [yellow]! {payload['message']}[/]")
        elif name in ("baseline_error", "submit_error", "submit_skipped"):
            emit(f"  [red]{name}: {payload}[/]")
        elif name == "submitting":
            emit(f"  submitting: {payload['cmd']}")
        elif name == "submitted":
            emit(f"  submitted: ok={payload['ok']}")
        elif name == "run_end":
            emit(f"[bold]■ done[/] — {payload['reason']}")

    return on_event


def _print_report(report, *, console_print) -> None:
    console_print("\n[bold]===== run report =====[/]")
    console_print(f"problem: {report.problem}")
    console_print(f"stopped: {report.stopped_reason}")
    if report.best and report.best.outcome.passed:
        o = report.best.outcome
        spd = f"{o.speedup:.2f}x reference" if o.speedup else "n/a"
        console_print(f"best: {report.best.id} ({report.best.approach}) — {spd}")
        console_print(f"  submission: {report.best.workspace / 'submission.py'}")
        if not report.submitted:
            console_print(
                f"  to submit: popcorn submit --leaderboard {report.problem} "
                f"--gpu <gpu> --mode leaderboard "
                f"{report.best.workspace / 'submission.py'}"
            )
    else:
        console_print("best: [red]no passing kernel found[/]")
    console_print(f"submitted: {report.submitted}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        return _cmd_list()
    if args.command == "seed-library":
        return _cmd_seed(args)
    if args.command == "run":
        return _cmd_run(args)
    return 2


def _cmd_seed(args: argparse.Namespace) -> int:
    from .library import Library
    from .seeds import seed_library

    lib = Library(args.library_dir)
    n = seed_library(lib)
    print(f"seeded {n} entries into {args.library_dir} ({len(lib)} total, index: {lib.index_kind})")
    return 0


def _cmd_list() -> int:
    try:
        with ProblemFetcher() as f:
            names = f.list_problems()
    except Exception as exc:  # network
        print(f"error listing problems: {exc}", file=sys.stderr)
        return 1
    for n in names:
        print(n)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    # Resolve stop conditions early so a bad spec fails fast.
    try:
        StoppingConditions.parse(args.stop_on)
    except StoppingConditionError as exc:
        print(f"invalid --stop-on: {exc}", file=sys.stderr)
        return 2

    hardware = HardwareProfile(args.hardware) if args.hardware else detect_hardware_profile()
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    auth_mode = AuthMode(args.auth_mode)
    if auth_mode is AuthMode.API_KEY and not api_key:
        print("--auth-mode api_key requires ANTHROPIC_API_KEY in the environment", file=sys.stderr)
        return 2

    # Resolve the problem once (default GPU) and reuse the fetcher (cached).
    fetcher = ProblemFetcher()
    try:
        problem = fetcher.fetch(args.leaderboard)
    except ProblemNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        fetcher.close()
        return 1
    except Exception as exc:  # network
        print(f"error fetching problem: {exc}", file=sys.stderr)
        fetcher.close()
        return 1

    gpu = args.gpu or problem.default_gpu()
    config = build_run_config(args, gpu=gpu, hardware=hardware, api_key=api_key)

    # Import the orchestrator lazily so `list` and arg errors don't pay for it.
    from .orchestrator import Orchestrator

    on_event = _make_event_printer()
    library = None
    if not args.no_library:
        try:
            from .library import Library

            library = Library(args.library_dir)
            if not args.tui:
                on_event("notice", {"message": f"library: {len(library)} entries ({library.index_kind})"})
        except Exception as exc:  # noqa: BLE001
            on_event("notice", {"message": f"library disabled: {exc}"})

    orch = Orchestrator(
        config, fetcher=fetcher, on_event=on_event,
        runs_dir=_path(args.runs_dir), library=library,
    )

    # TUI path: the dashboard owns the run (sets its own event handler) and blocks
    # until quit. We can't easily recover the report afterward, so report printing
    # is the headless path's job.
    if args.tui:
        from .tui import run_tui

        try:
            run_tui(orch, problem=config.leaderboard, gpu=gpu)
        finally:
            fetcher.close()
        return 0

    try:
        report = asyncio.run(orch.run())
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    finally:
        fetcher.close()

    try:
        from rich.console import Console

        _print_report(report, console_print=Console().print)
    except ImportError:
        _print_report(report, console_print=print)
    return 0 if (report.best and report.best.outcome.passed) else 1


def _path(p: str):
    from pathlib import Path

    return Path(p)


if __name__ == "__main__":
    raise SystemExit(main())
