"""Tests for the stuck-detector + clean-context research agent.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_research
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kernel_harness.agent import SubagentResult
from kernel_harness import subagents as sa
from kernel_harness.orchestrator import (
    CandidateRecord, EvalOutcome, IterationRecord, detect_stuck,
)
from kernel_harness.problem import Problem


# --------------------------------------------------------------------------- #
def _iter(index: int, *, speedup: float | None, passed: bool = True) -> IterationRecord:
    rec = CandidateRecord(
        id=f"c{index}", approach="triton", workspace=Path("."),
        outcome=EvalOutcome(passed=passed, speedup=speedup, geomean_ns=100.0 if passed else None),
    )
    return IterationRecord(index=index, candidates=[rec], best_id=rec.id)


def test_detect_stuck() -> None:
    # Steady improvement -> not stuck.
    improving = [_iter(i, speedup=1.0 + 0.3 * i) for i in range(6)]
    assert detect_stuck(improving) is None

    # Flat speedup over the window -> plateau.
    flat = [_iter(i, speedup=2.0) for i in range(6)]
    assert "plateau" in (detect_stuck(flat) or "")

    # Fewer than window+1 -> no plateau yet.
    assert detect_stuck([_iter(i, speedup=2.0) for i in range(3)]) is None

    # Three consecutive failing iterations -> correctness wall.
    walls = [_iter(0, speedup=1.5)] + [_iter(i, speedup=None, passed=False) for i in range(1, 4)]
    assert "correctness wall" in (detect_stuck(walls) or "")
    print("ok  test_detect_stuck")


def test_research_plan_from_dict() -> None:
    plan = sa.ResearchPlan.from_dict({
        "diagnosis": "memory-bound, compute being tuned",
        "strategy": "pivot",
        "actions": [{"what": "add split-K", "why": "grid is SM-starved"}, "use float4 loads"],
        "do_not_try": ["num_warps sweep (exp 3-5)"],
    })
    assert plan.strategy == "pivot"
    assert plan.actions[0] == "add split-K (grid is SM-starved)"
    assert plan.actions[1] == "use float4 loads"
    assert plan.focus == "add split-K (grid is SM-starved)"
    assert plan.do_not_try == ["num_warps sweep (exp 3-5)"]
    print("ok  test_research_plan_from_dict")


def test_research_runner_and_threading() -> None:
    reply = ('```json\n{"diagnosis":"wrong bottleneck — memory bound","strategy":"pivot",'
             '"actions":["add split-K flash-decoding"],"do_not_try":["tile sweep"]}\n```')

    class FakeRunner:
        def __init__(self): self.calls = []
        async def run_subagent(self, name, prompt, **kw):
            self.calls.append({"name": name, "prompt": prompt, **kw})
            return SubagentResult(name=name, model="m", text=reply, thinking="")

    runner = FakeRunner()
    brief = sa.ProblemBrief(summary="add", dtype="float16")
    plan = asyncio.run(
        sa.research(runner, brief, "ledger summary text", "def custom_kernel(d): ...", ["lesson: split-K"])
    )
    assert plan.strategy == "pivot"
    assert plan.actions == ["add split-K flash-decoding"]
    call = runner.calls[0]
    assert call["name"] == "research" and call["allowed_tools"] == []
    assert "ledger summary text" in call["prompt"]
    assert "def custom_kernel" in call["prompt"]
    assert "lesson: split-K" in call["prompt"]

    # plan threads into both the writer and reflection prompts
    wp = sa.build_write_prompt(brief, approach="triton", research=plan)
    assert "add split-K flash-decoding" in wp and "do NOT try" in wp
    rp = sa.build_reflect_prompt(brief, "history", plan)
    assert "research diagnosis" in rp and "wrong bottleneck" in rp
    print("ok  test_research_runner_and_threading")


# --------------------------------------------------------------------------- #
def test_orchestrator_fires_research_on_plateau() -> None:
    from kernel_harness.config import (
        AuthMode, BillingConfig, HardwareProfile, RunConfig, StoppingConditions,
    )
    from kernel_harness.orchestrator import Orchestrator, compute_speedup

    replies = {
        "problem_understander": '```json\n{"summary":"add","dtype":"float16"}\n```',
        "workload_inspector": '```json\n{"summary":"x","shortcuts":[]}\n```',
        "kernel_writer": '```json\n{"approach":"triton","summary":"k","local_test_passed":true}\n```',
        "profiler_interpreter": '```json\n{"bottleneck":"bandwidth","recommendations":["x"]}\n```',
        "reflection": '```json\n{"action":"iterate","reasoning":"go","next_approach":"triton","focus":"f"}\n```',
        "research": '```json\n{"diagnosis":"WRONG-BOTTLENECK-MARKER","strategy":"pivot",'
                    '"actions":["add split-K"],"do_not_try":["tile sweep"]}\n```',
    }

    class FakeRunner:
        def __init__(self): self.calls = []
        async def run_subagent(self, name, prompt, **kw):
            self.calls.append({"name": name, "prompt": prompt})
            return SubagentResult(name=name, model="m", text=replies[name], thinking="")

    class FakeFetcher:
        def fetch(self, name):  # noqa: ARG002
            return Problem(name="vectoradd_v2", problem_set="pmpp_v2", directory="d",
                           gpus=["B200"], spec_files={"task.yml": "x"}, submission_filename="submission.py")

    with tempfile.TemporaryDirectory() as td:
        cfg = RunConfig(leaderboard="vectoradd_v2", gpu="B200",
                        stop_on=StoppingConditions.parse("7iter"),
                        hardware=HardwareProfile.BLACKWELL,
                        billing=BillingConfig(mode=AuthMode.SUBSCRIPTION),
                        candidate_approaches=["triton"])
        events = []
        orch = Orchestrator(cfg, runner=(fr := FakeRunner()), fetcher=FakeFetcher(),
                            runs_dir=Path(td), on_event=lambda n, p: events.append((n, p)))

        async def fake_baseline(problem):  # noqa: ARG001
            return 1000.0

        def fake_evaluate(record, problem, ref):  # noqa: ARG001
            return EvalOutcome(passed=True, geomean_ns=500.0, speedup=compute_speedup(ref, 500.0))

        orch._establish_baseline = fake_baseline  # type: ignore[assignment]
        orch._evaluate = fake_evaluate  # type: ignore[assignment]
        asyncio.run(orch.run())

    names = [c["name"] for c in fr.calls]
    # flat speedup across iterations -> plateau -> research fired
    assert "research" in names, names
    assert any(n == "research" for n, _ in [(e[0], e[1]) for e in events]) or \
        any(e[0] == "research" for e in events), [e[0] for e in events]
    # the research diagnosis reached a later kernel_writer prompt
    research_idx = names.index("research")
    later_writer = [c for c in fr.calls[research_idx:] if c["name"] == "kernel_writer"]
    assert any("WRONG-BOTTLENECK-MARKER" in c["prompt"] for c in later_writer), "research plan didn't reach the rewrite"
    print("ok  test_orchestrator_fires_research_on_plateau")


if __name__ == "__main__":
    test_detect_stuck()
    test_research_plan_from_dict()
    test_research_runner_and_threading()
    test_orchestrator_fires_research_on_plateau()
    print("\nall research checks passed")
