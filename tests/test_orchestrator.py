"""Offline integration tests for the orchestrator.

The AgentRunner is faked (canned JSON → the real subagents run), and the
GPU-touching boundaries (``_establish_baseline``, ``_evaluate``, ``_maybe_submit``)
are stubbed per-instance. This exercises the actual loop control flow — parallel
candidates, ground-truth selection across iterations, stopping conditions, and
the reflection-driven submit/stop path — with no model and no GPU.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_orchestrator
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kernel_harness import orchestrator as orch_mod
from kernel_harness.agent import SubagentResult
from kernel_harness.config import (
    AuthMode,
    BillingConfig,
    HardwareProfile,
    RunConfig,
    StoppingConditions,
)
from kernel_harness.orchestrator import (
    CandidateRecord,
    EvalOutcome,
    IterationRecord,
    Orchestrator,
    RunReport,
    compute_speedup,
    render_history,
    select_best,
)
from kernel_harness.problem import Problem


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_compute_speedup() -> None:
    assert compute_speedup(1000.0, 500.0) == 2.0
    assert compute_speedup(1000.0, 0) is None
    assert compute_speedup(None, 500.0) is None
    assert compute_speedup(1000.0, None) is None
    print("ok  test_compute_speedup")


def _rec(cid: str, *, passed: bool, geomean: float | None) -> CandidateRecord:
    return CandidateRecord(
        id=cid, approach="x", workspace=Path("."),
        outcome=EvalOutcome(passed=passed, geomean_ns=geomean),
    )


def test_select_best() -> None:
    best = select_best([
        _rec("a", passed=True, geomean=1200.0),
        _rec("b", passed=True, geomean=800.0),
        _rec("c", passed=False, geomean=100.0),  # fast but wrong -> ignored
    ])
    assert best.id == "b", best.id

    # No timing -> first correct.
    assert select_best([_rec("a", passed=True, geomean=None)]).id == "a"
    # None correct -> None.
    assert select_best([_rec("c", passed=False, geomean=1.0)]) is None
    print("ok  test_select_best")


def test_render_history() -> None:
    it = IterationRecord(index=1, best_id="b", candidates=[
        _rec("a", passed=True, geomean=1200.0),
        _rec("b", passed=True, geomean=800.0),
    ])
    it.candidates[1].outcome.speedup = 1.25
    text = render_history([it], 1000.0)
    assert "reference_ns: 1000.0" in text
    assert "<-best" in text
    assert "iteration 1" in text
    print("ok  test_render_history")


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
class FakeRunner:
    def __init__(self, replies: dict[str, str]) -> None:
        self._replies = replies
        self.calls: list[str] = []

    async def run_subagent(self, name, prompt, **kwargs):
        self.calls.append(name)
        return SubagentResult(name=name, model="m", text=self._replies[name], thinking="")


_REPLIES = {
    "problem_understander": '```json\n{"summary":"add","dtype":"float16","kernel_signature":"custom_kernel(d)"}\n```',
    "kernel_writer": '```json\n{"approach":"triton","summary":"tiled","local_test_passed":true}\n```',
    "profiler_interpreter": '```json\n{"bottleneck":"bandwidth","evidence":"DRAM 95%","recommendations":["vectorize"]}\n```',
    "reflection": '```json\n{"action":"iterate","reasoning":"can go faster","next_approach":"triton","focus":"vectorize loads"}\n```',
}


def _problem() -> Problem:
    return Problem(
        name="vectoradd_v2", problem_set="pmpp_v2", directory="pmpp_v2/vectoradd_py",
        gpus=["B200"], spec_files={"task.yml": "description: add"},
        submission_filename="submission.py",
    )


def _config(stop: str, **kw) -> RunConfig:
    return RunConfig(
        leaderboard="vectoradd_v2", gpu="B200",
        stop_on=StoppingConditions.parse(stop),
        hardware=HardwareProfile.BLACKWELL,
        billing=BillingConfig(mode=AuthMode.SUBSCRIPTION),
        candidate_approaches=["triton", "cuda-inline", "pytorch"],
        **kw,
    )


class _FakeFetcher:
    def fetch(self, name):  # noqa: ARG002
        return _problem()


def _install_fakes(orch: Orchestrator, geomeans: dict[str, float], reference_ns: float = 1000.0):
    """Stub the GPU boundaries with canned numbers driven by candidate approach."""

    async def fake_baseline(problem):  # noqa: ARG001
        return reference_ns

    def fake_evaluate(record, problem, ref):  # noqa: ARG001
        g = geomeans.get(record.approach, 1500.0)
        # rewrite candidates get a distinct, faster number to test best-update
        if record.id.startswith("rw-"):
            g = 500.0
        return EvalOutcome(
            passed=True, geomean_ns=g, speedup=compute_speedup(ref, g), source="local",
        )

    orch._establish_baseline = fake_baseline  # type: ignore[assignment]
    orch._evaluate = fake_evaluate  # type: ignore[assignment]


def test_loop_stops_on_iteration_budget() -> None:
    with tempfile.TemporaryDirectory() as td:
        cfg = _config("2iter")
        orch = Orchestrator(
            cfg, runner=FakeRunner(_REPLIES), fetcher=_FakeFetcher(),
            runs_dir=Path(td),
        )
        _install_fakes(orch, {"triton": 800.0, "cuda-inline": 1200.0, "pytorch": 2000.0})
        report: RunReport = asyncio.run(orch.run())

    assert "iteration budget" in report.stopped_reason, report.stopped_reason
    assert len(report.iterations) == 2, len(report.iterations)
    # iter1 best = triton (800); iter2 rewrite = 500 -> new best, 2x speedup
    assert report.best is not None
    assert report.best.id.startswith("rw-"), report.best.id
    assert abs(report.best.outcome.speedup - 2.0) < 1e-6, report.best.outcome.speedup
    print("ok  test_loop_stops_on_iteration_budget")


def test_loop_stops_on_target_speedup() -> None:
    with tempfile.TemporaryDirectory() as td:
        cfg = _config("5x_reference")  # iter1 triton=800 -> 1.25x, not enough
        orch = Orchestrator(
            cfg, runner=FakeRunner(_REPLIES), fetcher=_FakeFetcher(), runs_dir=Path(td),
        )
        # Make iter1 triton hit 5x immediately (geomean 200 -> 1000/200=5x).
        _install_fakes(orch, {"triton": 200.0, "cuda-inline": 1200.0, "pytorch": 2000.0})
        report = asyncio.run(orch.run())

    assert "target speedup" in report.stopped_reason, report.stopped_reason
    assert len(report.iterations) == 1  # hit target after iteration 1
    assert report.best.approach == "triton"
    print("ok  test_loop_stops_on_target_speedup")


def test_reflection_submit_breaks_and_submits() -> None:
    replies = dict(_REPLIES)
    replies["reflection"] = '```json\n{"action":"submit","reasoning":"plateaued at good speed"}\n```'
    submitted_flag = {"v": False}

    with tempfile.TemporaryDirectory() as td:
        cfg = _config("20iter")  # high budget; reflection should end it at iter2
        orch = Orchestrator(
            cfg, runner=FakeRunner(replies), fetcher=_FakeFetcher(), runs_dir=Path(td),
        )
        _install_fakes(orch, {"triton": 800.0, "cuda-inline": 1200.0, "pytorch": 2000.0})

        async def fake_submit(problem, best, report):  # noqa: ARG001
            submitted_flag["v"] = True
            report.submitted = True

        orch._maybe_submit = fake_submit  # type: ignore[assignment]
        report = asyncio.run(orch.run())

    assert "submit" in report.stopped_reason, report.stopped_reason
    assert submitted_flag["v"], "reflection 'submit' should trigger _maybe_submit"
    assert report.submitted
    # iter2 broke before producing a rewrite candidate, so best stays iter1 triton
    assert report.best.approach == "triton"
    print("ok  test_reflection_submit_breaks_and_submits")


if __name__ == "__main__":
    test_compute_speedup()
    test_select_best()
    test_render_history()
    test_loop_stops_on_iteration_budget()
    test_loop_stops_on_target_speedup()
    test_reflection_submit_breaks_and_submits()
    print("\nall orchestrator checks passed")
