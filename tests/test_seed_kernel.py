"""Tests for the --seed-kernel warm-start.

Encodes the user's "competing candidate / structural reference, never an anchor"
requirement: a passing seed must be beatable by a fresh candidate, and a failing
seed must flow to the writers as a reference (not poison the pool).

Run: PYTHONPATH=src .venv/bin/python -m tests.test_seed_kernel
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kernel_harness.agent import SubagentResult
from kernel_harness import subagents as sa
from kernel_harness.cli import build_parser, build_run_config
from kernel_harness.config import (
    AuthMode, BillingConfig, HardwareProfile, RunConfig, StoppingConditions,
)
from kernel_harness.orchestrator import EvalOutcome, Orchestrator, compute_speedup
from kernel_harness.problem import Problem


def test_build_write_prompt_seed_reference() -> None:
    brief = sa.ProblemBrief(summary="add", dtype="float16")
    p = sa.build_write_prompt(brief, approach="triton", seed_reference="def custom_kernel(d): return d  # SEEDMARK")
    assert "STRUCTURAL REFERENCE" in p and "SEEDMARK" in p
    # absent when no seed
    assert "STRUCTURAL REFERENCE" not in sa.build_write_prompt(brief, approach="triton")
    print("ok  test_build_write_prompt_seed_reference")


def test_cli_seed_flag() -> None:
    args = build_parser().parse_args(["run", "-l", "vectoradd_v2", "--seed-kernel", "/tmp/my_kernel.py"])
    cfg = build_run_config(args, gpu="B200", hardware=HardwareProfile.BLACKWELL, api_key=None)
    assert cfg.seed_kernel == Path("/tmp/my_kernel.py")
    # default: none
    args2 = build_parser().parse_args(["run", "-l", "x"])
    assert build_run_config(args2, gpu="B200", hardware=HardwareProfile.BLACKWELL, api_key=None).seed_kernel is None
    print("ok  test_cli_seed_flag")


_REPLIES = {
    "problem_understander": '```json\n{"summary":"add","dtype":"float16"}\n```',
    "workload_inspector": '```json\n{"summary":"x","shortcuts":[]}\n```',
    "kernel_writer": '```json\n{"approach":"triton","summary":"k","local_test_passed":true}\n```',
}


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_subagent(self, name, prompt, **kw):
        self.calls.append({"name": name, "prompt": prompt})
        return SubagentResult(name=name, model="m", text=_REPLIES[name], thinking="")


class FakeFetcher:
    def fetch(self, name):  # noqa: ARG002
        return Problem(name="vectoradd_v2", problem_set="pmpp_v2", directory="d",
                       gpus=["B200"], spec_files={"task.yml": "x"}, submission_filename="submission.py")


def _orch(td: str, seed_path: Path, runner: FakeRunner, events: list):
    cfg = RunConfig(
        leaderboard="vectoradd_v2", gpu="B200",
        stop_on=StoppingConditions.parse("1iter"),  # iteration 1 only
        hardware=HardwareProfile.BLACKWELL,
        billing=BillingConfig(mode=AuthMode.SUBSCRIPTION),
        candidate_approaches=["triton"],
        seed_kernel=seed_path,
    )
    orch = Orchestrator(cfg, runner=runner, fetcher=FakeFetcher(), runs_dir=Path(td),
                        on_event=lambda n, p: events.append((n, p)))

    async def fake_baseline(problem):  # noqa: ARG001
        return 1000.0

    orch._establish_baseline = fake_baseline  # type: ignore[assignment]
    return orch


def test_passing_seed_competes_but_loses_to_faster_candidate() -> None:
    """Anti-anchor: a passing seed is in the pool, but a faster cold candidate wins."""
    def fake_evaluate(record, problem, ref):  # noqa: ARG001
        g = 900.0 if record.id == "seed" else 500.0  # cold triton is faster
        return EvalOutcome(passed=True, geomean_ns=g, speedup=compute_speedup(ref, g), source="local")

    with tempfile.TemporaryDirectory() as td:
        seed = Path(td) / "seed.py"
        seed.write_text("def custom_kernel(d):\n    return d  # SLOW_BUT_CORRECT\n")
        events: list = []
        orch = _orch(td, seed, FakeRunner(), events)
        orch._evaluate = fake_evaluate  # type: ignore[assignment]
        report = asyncio.run(orch.run())

    # seed was evaluated as a competing candidate ...
    evald = {p["id"] for n, p in events if n == "candidate_evaluated"}
    assert "seed" in evald, evald
    assert any(n == "seed" and p.get("status") == "pass" for n, p in events)
    # ... but the faster fresh candidate won — NOT anchored to the seed
    assert report.best is not None and report.best.approach == "triton", report.best.approach
    assert abs(report.best.outcome.speedup - 2.0) < 1e-6
    print("ok  test_passing_seed_competes_but_loses_to_faster_candidate")


def test_failing_seed_becomes_structural_reference() -> None:
    """A seed that fails correctness is handed to the writers as a reference, not pooled."""
    def fake_evaluate(record, problem, ref):  # noqa: ARG001
        if record.id == "seed":
            return EvalOutcome(passed=False, error="wrong output", source="local")
        return EvalOutcome(passed=True, geomean_ns=500.0, speedup=compute_speedup(ref, 500.0), source="local")

    with tempfile.TemporaryDirectory() as td:
        seed = Path(td) / "seed.py"
        seed.write_text("def custom_kernel(d):\n    return None  # BROKEN_SEEDMARK\n")
        events: list = []
        runner = FakeRunner()
        orch = _orch(td, seed, runner, events)
        orch._evaluate = fake_evaluate  # type: ignore[assignment]
        report = asyncio.run(orch.run())

    assert any(n == "seed" and p.get("status") == "reference" for n, p in events), events
    # the failing seed's code reached the cold writer as a structural reference
    writer_prompts = [c["prompt"] for c in runner.calls if c["name"] == "kernel_writer"]
    assert writer_prompts and any("BROKEN_SEEDMARK" in p for p in writer_prompts)
    assert any("STRUCTURAL REFERENCE" in p for p in writer_prompts)
    # seed did NOT enter the pool (only the cold candidate did)
    evald = {p["id"] for n, p in events if n == "candidate_evaluated"}
    assert "seed" not in evald, evald
    assert report.best.approach == "triton"
    print("ok  test_failing_seed_becomes_structural_reference")


if __name__ == "__main__":
    test_build_write_prompt_seed_reference()
    test_cli_seed_flag()
    test_passing_seed_competes_but_loses_to_faster_candidate()
    test_failing_seed_becomes_structural_reference()
    print("\nall seed-kernel checks passed")
