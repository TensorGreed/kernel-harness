"""Tests for the persistent on-disk experiment ledger.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_ledger
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kernel_harness.ledger import Ledger
from kernel_harness.orchestrator import CandidateRecord, EvalOutcome, IterationRecord
from kernel_harness.subagents import KernelCandidate, ProblemBrief, ProfilerFindings, StrategyDecision, WorkloadProfile


def _cand(cid: str, approach: str, *, passed: bool, geomean: float | None, speedup: float | None, ws: Path) -> CandidateRecord:
    return CandidateRecord(
        id=cid, approach=approach, workspace=ws,
        candidate=KernelCandidate(approach=approach, summary=f"{approach} kernel"),
        outcome=EvalOutcome(passed=passed, geomean_ns=geomean, speedup=speedup, source="local",
                            error=None if passed else "mismatch"),
    )


def test_header_and_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        led = Ledger(root)
        led.write_header(
            problem="vectoradd_v2",
            brief=ProblemBrief(summary="add", dtype="float16", kernel_signature="custom_kernel(d)",
                               optimization_targets=["bandwidth"]),
            gpu="B200", reference_ns=1000.0,
            workload=WorkloadProfile(summary="SM-starved", shortcuts=["split-K"]),
        )
        text = led.render()
        assert "# Run — vectoradd_v2" in text
        assert "reference baseline (ns): 1000.0" in text
        assert "shortcut: split-K" in text
        assert "| Iter | Candidate |" in text

        c1 = _cand("i1-c0-triton", "triton", passed=True, geomean=800.0, speedup=1.25, ws=root / "c0")
        c2 = _cand("i1-c1-cuda", "cuda", passed=False, geomean=None, speedup=None, ws=root / "c1")
        led.record_candidate(iteration=1, record=c1, is_best=True)
        led.record_candidate(iteration=1, record=c2, is_best=False)

        text = led.render()
        assert "| 1 | i1-c0-triton | triton | pass | 800.0 | 1.25x | **best** |" in text, text
        assert "| 1 | i1-c1-cuda | cuda | FAIL |" in text

        # per-candidate result.md written into the workspace
        r1 = (root / "c0" / "result.md").read_text()
        assert "approach: triton" in r1 and "speedup: 1.25x" in r1
        assert "writer summary: triton kernel" in r1
        r2 = (root / "c1" / "result.md").read_text()
        assert "error: mismatch" in r2
        print("ok  test_header_and_rows")


def test_iteration_note() -> None:
    with tempfile.TemporaryDirectory() as td:
        led = Ledger(Path(td))
        led.write_header(problem="p", brief=None, gpu="B200", reference_ns=None, workload=None)
        it = IterationRecord(index=2, profiler=ProfilerFindings(bottleneck="memory bandwidth"),
                             decision=StrategyDecision(action="iterate", focus="vectorize loads"))
        led.record_iteration_note(2, it)
        text = led.render()
        assert "profiler: memory bandwidth" in text
        assert "decision: iterate — vectorize loads" in text
        print("ok  test_iteration_note")


def test_render_empty_safe() -> None:
    with tempfile.TemporaryDirectory() as td:
        led = Ledger(Path(td) / "sub")  # header never written
        assert led.render() == ""  # no crash, empty
        print("ok  test_render_empty_safe")


def test_orchestrator_writes_ledger() -> None:
    """A real (faked) run leaves a summary.md with candidate rows + best marker."""
    from kernel_harness.agent import SubagentResult
    from kernel_harness.config import AuthMode, BillingConfig, HardwareProfile, RunConfig, StoppingConditions
    from kernel_harness.orchestrator import Orchestrator, compute_speedup
    from kernel_harness.problem import Problem

    replies = {
        "problem_understander": '```json\n{"summary":"add","dtype":"float16"}\n```',
        "workload_inspector": '```json\n{"summary":"sm-starved","shortcuts":["split-K"]}\n```',
        "kernel_writer": '```json\n{"approach":"triton","summary":"tiled","local_test_passed":true}\n```',
        "profiler_interpreter": '```json\n{"bottleneck":"bandwidth","recommendations":["x"]}\n```',
        "reflection": '```json\n{"action":"stop","reasoning":"good"}\n```',
    }

    class FakeRunner:
        async def run_subagent(self, name, prompt, **kw):
            return SubagentResult(name=name, model="m", text=replies[name], thinking="")

    class FakeFetcher:
        def fetch(self, name):  # noqa: ARG002
            return Problem(name="vectoradd_v2", problem_set="pmpp_v2", directory="d",
                           gpus=["B200"], spec_files={"task.yml": "x"}, submission_filename="submission.py")

    with tempfile.TemporaryDirectory() as td:
        cfg = RunConfig(leaderboard="vectoradd_v2", gpu="B200",
                        stop_on=StoppingConditions.parse("3iter"),
                        hardware=HardwareProfile.BLACKWELL,
                        billing=BillingConfig(mode=AuthMode.SUBSCRIPTION),
                        candidate_approaches=["triton", "cuda"])
        orch = Orchestrator(cfg, runner=FakeRunner(), fetcher=FakeFetcher(), runs_dir=Path(td))

        async def fake_baseline(problem):  # noqa: ARG001
            return 1000.0

        def fake_evaluate(record, problem, ref):  # noqa: ARG001
            g = 800.0 if record.approach == "triton" else 1500.0
            return EvalOutcome(passed=True, geomean_ns=g, speedup=compute_speedup(ref, g), source="local")

        orch._establish_baseline = fake_baseline  # type: ignore[assignment]
        orch._evaluate = fake_evaluate  # type: ignore[assignment]
        asyncio.run(orch.run())

        # exactly one run dir, with a populated summary.md
        run_dirs = [p for p in Path(td).iterdir() if p.is_dir()]
        assert len(run_dirs) == 1, run_dirs
        summary = (run_dirs[0] / "summary.md").read_text()
        assert "# Run — vectoradd_v2" in summary
        assert "shortcut: split-K" in summary           # workload header persisted
        assert "triton" in summary and "**best**" in summary
        assert "| 1 |" in summary                        # iteration-1 rows present
        print("ok  test_orchestrator_writes_ledger")


if __name__ == "__main__":
    test_header_and_rows()
    test_iteration_note()
    test_render_empty_safe()
    test_orchestrator_writes_ledger()
    print("\nall ledger checks passed")
