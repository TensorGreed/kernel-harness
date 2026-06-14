"""The orchestrator — the deterministic loop that drives the subagents.

This is the integration keystone. It owns control flow; the subagents are its
moving parts. One run:

1. Fetch the problem; ``understand_problem`` → brief.
2. Establish a **reference baseline** by benchmarking ``ref_kernel`` locally
   (``from reference import ref_kernel as custom_kernel``). Speedup is measured
   against this.
3. ``retrieve_knowledge`` from the library (candidate notes; empty until the
   library is built).
4. **Iteration 1** — write several candidates *in parallel* (one per approach),
   each in its own isolated workspace + tool server, then evaluate each for
   ground-truth correctness/timing **serially** (clean numbers on one GPU).
5. **Iteration 2+** — ``interpret_profile`` the best, ``reflect`` to decide
   iterate / submit / stop, then a focused ``write_kernel`` rewrite addressing
   the profiler's bottleneck. Re-evaluate; keep the best.
6. After each iteration, check ``StoppingConditions`` (time / iterations /
   target speedup) and the reflection decision and an external stop event (TUI).
7. Report the best kernel; optionally fire a ranked popcorn submission
   (``auto_submit``, off by default — outward-facing); ``update_library``.

Ground-truth evaluation goes through ``evalproto`` directly (not the agent's
self-report) so the loop trusts measured numbers. Real model calls + GB10 tools
happen here; the decision logic is unit-tested with everything faked.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import evalproto, subagents
from .agent import AgentRunner, CreditExhaustedError
from .config import HardwareProfile, RunConfig
from .ledger import Ledger
from .problem import Problem, ProblemFetcher
from .subagents import (
    KernelCandidate,
    ProblemBrief,
    ProfilerFindings,
    ResearchPlan,
    RetrievedKnowledge,
    StrategyDecision,
    WorkloadProfile,
)
from .tools import RunContext, build_popcorn_cmd, build_tool_server

# A submission that benchmarks the problem's own reference implementation.
REFERENCE_SUBMISSION = "from reference import ref_kernel as custom_kernel\n"

# Turn budget for the tool-using kernel-writer / profiler subagents.
_WRITER_MAX_TURNS = 50
_PROFILER_MAX_TURNS = 20


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class EvalOutcome:
    """Ground-truth evaluation of a candidate."""

    passed: bool = False
    geomean_ns: float | None = None
    speedup: float | None = None        # reference_ns / geomean_ns
    source: str = "local"               # "local" | "agent-claim"
    detail: str = ""
    error: str | None = None


@dataclass
class CandidateRecord:
    id: str
    approach: str
    workspace: Path
    candidate: KernelCandidate | None = None
    outcome: EvalOutcome = field(default_factory=EvalOutcome)


@dataclass
class IterationRecord:
    index: int
    candidates: list[CandidateRecord] = field(default_factory=list)
    best_id: str | None = None
    profiler: ProfilerFindings | None = None
    decision: StrategyDecision | None = None
    research: ResearchPlan | None = None
    trigger: str | None = None       # why research was fired this iteration


@dataclass
class RunReport:
    problem: str
    reference_ns: float | None
    iterations: list[IterationRecord] = field(default_factory=list)
    best: CandidateRecord | None = None
    stopped_reason: str = ""
    submitted: bool = False
    brief: ProblemBrief | None = None
    workload: WorkloadProfile | None = None


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def compute_speedup(reference_ns: float | None, geomean_ns: float | None) -> float | None:
    """Speedup of a candidate over the reference (higher = faster)."""
    if not reference_ns or not geomean_ns or geomean_ns <= 0:
        return None
    return reference_ns / geomean_ns


def select_best(records: list[CandidateRecord]) -> CandidateRecord | None:
    """Pick the correct candidate with the lowest geomean time.

    Correct candidates with a measured time win first; if none have timing
    (e.g. leaderboard-only), fall back to the first correct one; else None.
    """
    correct = [r for r in records if r.outcome.passed]
    if not correct:
        return None
    timed = [r for r in correct if r.outcome.geomean_ns is not None]
    if timed:
        return min(timed, key=lambda r: r.outcome.geomean_ns)  # type: ignore[arg-type]
    return correct[0]


def detect_stuck(
    iterations: list[IterationRecord],
    *,
    plateau_window: int = 4,
    plateau_pct: float = 0.05,
    fail_window: int = 3,
) -> str | None:
    """Return a trigger reason if the loop is stuck, else None (pure).

    Two robust, computable triggers — the cue to fire the expensive research
    agent instead of routine reflection:
    * **correctness wall** — the last ``fail_window`` iterations produced no
      passing candidate.
    * **plateau** — the running-best speedup improved < ``plateau_pct`` over the
      last ``plateau_window`` iterations.
    """
    if not iterations:
        return None

    # Correctness wall.
    if len(iterations) >= fail_window:
        recent = iterations[-fail_window:]
        if all(not any(c.outcome.passed for c in it.candidates) for it in recent):
            return f"correctness wall: no passing candidate in {fail_window} iterations"

    # Running-best speedup per iteration.
    best_seq: list[float] = []
    cur = 0.0
    for it in iterations:
        for c in it.candidates:
            if c.outcome.passed and c.outcome.speedup:
                cur = max(cur, c.outcome.speedup)
        best_seq.append(cur)

    if len(best_seq) > plateau_window:
        prev, now = best_seq[-1 - plateau_window], best_seq[-1]
        if prev > 0 and (now / prev - 1.0) < plateau_pct:
            return (
                f"plateau: best speedup improved <{plateau_pct:.0%} "
                f"over {plateau_window} iterations"
            )
    return None


def render_history(iterations: list[IterationRecord], reference_ns: float | None) -> str:
    """Compact, model-readable summary of the run so far for ``reflect``."""
    lines = [f"reference_ns: {reference_ns:.1f}" if reference_ns else "reference_ns: unknown"]
    for it in iterations:
        lines.append(f"iteration {it.index}:")
        for r in it.candidates:
            o = r.outcome
            spd = f"{o.speedup:.2f}x" if o.speedup else "n/a"
            t = f"{o.geomean_ns:.1f}ns" if o.geomean_ns else "n/a"
            tag = " <-best" if r.id == it.best_id else ""
            status = "pass" if o.passed else f"fail({o.error or 'incorrect'})"
            lines.append(f"  [{r.id}] {r.approach}: {status} time={t} speedup={spd}{tag}")
        if it.profiler:
            lines.append(f"  profiler: {it.profiler.bottleneck}")
        if it.decision:
            lines.append(f"  decision: {it.decision.action} ({it.decision.focus})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    """Drives one optimization run end to end."""

    def __init__(
        self,
        config: RunConfig,
        *,
        runner: AgentRunner | None = None,
        fetcher: ProblemFetcher | None = None,
        runs_dir: Path | None = None,
        library=None,
        library_candidates: Callable[[ProblemBrief], list[str]] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        max_iterations_safety: int = 50,
    ) -> None:
        self.config = config
        self._fetcher = fetcher
        self._runner = runner
        self._runs_dir = runs_dir or Path("runs")
        # A Library (if given) supplies retrieval candidates and receives the
        # run's distilled lessons; an explicit callback still overrides it.
        self._library = library
        if library_candidates is not None:
            self._library_candidates = library_candidates
        elif library is not None:
            self._library_candidates = library.candidates_for
        else:
            self._library_candidates = lambda _brief: []
        self._on_event = on_event or (lambda _name, _payload: None)
        self._max_iterations_safety = max_iterations_safety
        self._stop_event = asyncio.Event()
        self._run_id = time.strftime("%Y%m%d-%H%M%S")
        self._workload: WorkloadProfile | None = None
        self._ledger: Ledger | None = None

    def request_stop(self) -> None:
        """Signal the loop to stop after the current iteration (TUI 'force stop')."""
        self._stop_event.set()

    def set_on_event(self, callback: Callable[[str, dict], None]) -> None:
        """Replace the progress-event handler (used by the TUI to wire its display)."""
        self._on_event = callback

    def emit(self, name: str, **payload) -> None:
        self._on_event(name, payload)

    # ----------------------------------------------------------------- #
    async def run(self) -> RunReport:
        problem = self._get_problem()
        runner = self._runner or self._build_runner()
        self.emit("run_start", problem=problem.name, gpu=self.config.gpu)

        brief = await subagents.understand_problem(runner, problem)
        self.emit("brief", summary=brief.summary)

        # Analyze the input distribution for exploitable structure (split-K,
        # padding-exit, per-regime dispatch, trivializing shapes). Best-effort —
        # a failure here shouldn't abort the run.
        try:
            self._workload = await subagents.inspect_workload(runner, problem, brief)
            self.emit(
                "workload", summary=self._workload.summary,
                shortcuts=len(self._workload.shortcuts),
            )
        except Exception as exc:  # noqa: BLE001
            self.emit("workload_error", error=str(exc))
            self._workload = None

        reference_ns = await self._establish_baseline(problem)
        self.emit("baseline", reference_ns=reference_ns)

        knowledge = await subagents.retrieve_knowledge(
            runner, brief, self._library_candidates(brief)
        )

        report = RunReport(
            problem=problem.name, reference_ns=reference_ns, brief=brief,
            workload=self._workload,
        )

        # Persistent on-disk experiment ledger — the durable, inspectable record
        # that reflect/update_library read from disk (survives a crash; bounded
        # prompt size on long runs; substrate for a future research agent).
        self._ledger = Ledger(self._runs_dir / self._run_id)
        self._ledger.write_header(
            problem=problem.name, brief=brief, gpu=self.config.gpu,
            reference_ns=reference_ns, workload=self._workload,
        )

        best: CandidateRecord | None = None
        start = time.monotonic()
        it_index = 0

        while True:
            it_index += 1
            self.emit("iteration_start", index=it_index)
            it_record = IterationRecord(index=it_index)

            if it_index == 1:
                records = await self._iteration_one(runner, problem, brief, knowledge)
            else:
                # Profile the best, then decide, then (maybe) a focused rewrite.
                profiler = await self._profile_best(runner, problem, brief, best)
                it_record.profiler = profiler

                # Cheap per-iteration decision; fire the expensive clean-context
                # research agent ONLY when a plateau / correctness-wall trigger
                # fires — deep diagnosis when stuck, not every iteration.
                research_plan = None
                trigger = detect_stuck(report.iterations)
                if trigger:
                    it_record.trigger = trigger
                    self.emit("research", trigger=trigger)
                    research_plan = await self._run_research(runner, problem, brief, best)
                    it_record.research = research_plan

                decision = await subagents.reflect(
                    runner, brief, self._ledger.render(), research_plan
                )
                it_record.decision = decision
                self.emit("decision", action=decision.action, focus=decision.focus)
                if decision.action in ("submit", "stop"):
                    report.iterations.append(it_record)
                    self._ledger.record_iteration_note(it_index, it_record)
                    report.stopped_reason = f"reflection: {decision.action} — {decision.reasoning}"
                    if decision.action == "submit":
                        await self._maybe_submit(problem, best, report)
                    break
                record = await self._rewrite(
                    runner, problem, brief, knowledge, best, profiler, decision, research_plan
                )
                records = [record]

            # Authoritative, serial ground-truth evaluation.
            for r in records:
                r.outcome = self._evaluate(r, problem, reference_ns)
                self.emit(
                    "candidate_evaluated", id=r.id, approach=r.approach,
                    passed=r.outcome.passed, speedup=r.outcome.speedup,
                )

            pool = ([best] if best else []) + records
            best = select_best(pool) or best
            it_record.candidates = records
            it_record.best_id = best.id if best else None
            report.iterations.append(it_record)
            report.best = best

            # Persist this iteration to the ledger (candidates + any notes).
            for r in records:
                self._ledger.record_candidate(
                    iteration=it_index, record=r, is_best=bool(best and r.id == best.id)
                )
            self._ledger.record_iteration_note(it_index, it_record)

            # Stop checks: external (TUI), then declared conditions, then safety.
            if self._stop_event.is_set():
                report.stopped_reason = "manual stop requested"
                break
            reason = self.config.stop_on.met(
                elapsed_seconds=time.monotonic() - start,
                iterations=it_index,
                best_speedup=best.outcome.speedup if best else None,
            )
            if reason:
                report.stopped_reason = reason
                break
            if it_index >= self._max_iterations_safety:
                report.stopped_reason = f"safety iteration cap ({self._max_iterations_safety})"
                break

        if self.config.auto_submit and not report.submitted and best and best.outcome.passed:
            await self._maybe_submit(problem, best, report)

        await self._persist_lessons(runner, problem, brief, report)

        self.emit(
            "run_end", reason=report.stopped_reason,
            best=report.best.id if report.best else None,
            best_speedup=report.best.outcome.speedup if report.best else None,
        )
        return report

    # ----------------------------------------------------------------- #
    # Phases
    # ----------------------------------------------------------------- #
    def _build_runner(self):
        """Build the cloud runner, wrapped in a backend router if local is in use."""
        notice = lambda m: self.emit("notice", message=m)  # noqa: E731
        cloud = AgentRunner(self.config, on_notice=notice)
        if not self.config.uses_local():
            return cloud
        from .backends import BackendRouter, LocalRunner

        local = LocalRunner(self.config)
        self.emit("notice", message=f"local backend: {self.config.local.base_url}")
        return BackendRouter(cloud, local, self.config, on_notice=notice)

    def _get_problem(self) -> Problem:
        if self._fetcher is not None:
            return self._fetcher.fetch(self.config.leaderboard)
        with ProblemFetcher() as f:
            return f.fetch(self.config.leaderboard)

    async def _run_research(
        self,
        runner: AgentRunner,
        problem: Problem,
        brief: ProblemBrief,
        best: CandidateRecord | None,
    ) -> ResearchPlan | None:
        """Deep clean-context diagnosis from on-disk artifacts. Best-effort."""
        try:
            best_code = ""
            if best is not None:
                sub = best.workspace / problem.submission_filename
                if sub.exists():
                    best_code = sub.read_text()
            notes = self._library_candidates(brief)
            return await subagents.research(
                runner, brief, self._ledger.render() if self._ledger else "", best_code, notes
            )
        except Exception as exc:  # noqa: BLE001 — never abort a run on diagnosis
            self.emit("research_error", error=str(exc))
            return None

    async def _establish_baseline(self, problem: Problem) -> float | None:
        """Benchmark the reference implementation locally to anchor speedups."""
        if not self.config.hardware.can_run_locally:
            return None
        ws = self._workspace("baseline")
        try:
            evalproto.stage_workspace(ws, problem, REFERENCE_SUBMISSION)
            result = await asyncio.to_thread(
                evalproto.run_eval, ws, "benchmark", problem
            )
        except evalproto.LocalEvalError as exc:
            self.emit("baseline_error", error=str(exc))
            return None
        if not result.passed or result.geomean_ns is None:
            self.emit("baseline_error", error=result.error or "reference benchmark failed")
            return None
        return result.geomean_ns

    async def _iteration_one(
        self,
        runner: AgentRunner,
        problem: Problem,
        brief: ProblemBrief,
        knowledge: RetrievedKnowledge,
    ) -> list[CandidateRecord]:
        approaches = self.config.candidate_approaches or ["pytorch"]
        tasks = [
            self._write_candidate(runner, problem, brief, knowledge, approach, slot=i)
            for i, approach in enumerate(approaches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        records: list[CandidateRecord] = []
        for approach, res in zip(approaches, results):
            if isinstance(res, CreditExhaustedError):
                raise res  # out of ways to pay -> abort the whole run
            if isinstance(res, Exception):
                self.emit("candidate_failed", approach=approach, error=str(res))
                continue
            records.append(res)
        return records

    async def _write_candidate(
        self,
        runner: AgentRunner,
        problem: Problem,
        brief: ProblemBrief,
        knowledge: RetrievedKnowledge,
        approach: str,
        *,
        slot: int,
        prior_summary: str = "",
        profiler: ProfilerFindings | None = None,
    ) -> CandidateRecord:
        cid = f"i1-c{slot}-{approach}"
        ws = self._workspace(cid)
        ctx = self._context(problem, ws)
        servers, allowed = build_tool_server(ctx)
        full_allowed = allowed + ["Write", "Read", "Edit", "Bash"]
        candidate = await subagents.write_kernel(
            runner, brief,
            approach=approach,
            allowed_tools=full_allowed,
            mcp_servers=servers,
            knowledge=knowledge,
            gpu=self.config.gpu,
            hardware_notes=self._hardware_notes(),
            prior_summary=prior_summary,
            profiler=profiler,
            workload=self._workload,
        )
        return CandidateRecord(id=cid, approach=approach, workspace=ws, candidate=candidate)

    async def _profile_best(
        self,
        runner: AgentRunner,
        problem: Problem,
        brief: ProblemBrief,
        best: CandidateRecord | None,
    ) -> ProfilerFindings | None:
        if best is None or not self.config.hardware.can_profile:
            return None
        ctx = self._context(problem, best.workspace)
        servers, allowed = build_tool_server(ctx)
        return await subagents.interpret_profile(
            runner, brief,
            allowed_tools=allowed + ["Read", "Bash"],
            mcp_servers=servers,
            gpu=self.config.gpu,
            hardware_notes=self._hardware_notes(),
        )

    async def _rewrite(
        self,
        runner: AgentRunner,
        problem: Problem,
        brief: ProblemBrief,
        knowledge: RetrievedKnowledge,
        best: CandidateRecord | None,
        profiler: ProfilerFindings | None,
        decision: StrategyDecision,
        research_plan: ResearchPlan | None = None,
    ) -> CandidateRecord:
        approach = decision.next_approach or (best.approach if best else "pytorch")
        prior = best.candidate.summary if best and best.candidate else ""
        # Seed the rewrite workspace from the current best so the agent iterates
        # on working code rather than starting cold.
        cid = f"rw-{int(time.monotonic()*1000) % 100000}-{approach}"
        ws = self._workspace(cid)
        if best is not None:
            self._seed_workspace(ws, best.workspace, problem)
        ctx = self._context(problem, ws)
        servers, allowed = build_tool_server(ctx)
        candidate = await subagents.write_kernel(
            runner, brief,
            approach=approach,
            allowed_tools=allowed + ["Write", "Read", "Edit", "Bash"],
            mcp_servers=servers,
            knowledge=knowledge,
            gpu=self.config.gpu,
            hardware_notes=self._hardware_notes(),
            prior_summary=prior,
            profiler=profiler,
            workload=self._workload,
            research=research_plan,
        )
        return CandidateRecord(id=cid, approach=approach, workspace=ws, candidate=candidate)

    async def _maybe_submit(
        self, problem: Problem, best: CandidateRecord | None, report: RunReport
    ) -> None:
        if best is None or not best.outcome.passed:
            self.emit("submit_skipped", reason="no passing kernel")
            return
        cmd = build_popcorn_cmd(self._context(problem, best.workspace), "leaderboard")
        self.emit("submitting", cmd=" ".join(cmd))
        import subprocess

        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=1800
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            self.emit("submit_error", error=str(exc))
            return
        report.submitted = proc.returncode == 0
        self.emit("submitted", ok=report.submitted, output=(proc.stdout or "")[-2000:])

    async def _persist_lessons(
        self, runner: AgentRunner, problem: Problem, brief: ProblemBrief, report: RunReport
    ) -> None:
        """Distill the run via ``update_library`` and persist to the library.

        Best-effort: a library/subagent failure must not fail the run.
        """
        if self._library is None:
            return
        try:
            outcome = self._ledger.render() if self._ledger else render_history(
                report.iterations, report.reference_ns
            )
            if report.best:
                o = report.best.outcome
                outcome += f"\nbest: {report.best.approach} speedup={o.speedup} passed={o.passed}"
            entries = await subagents.update_library(runner, brief, outcome)
            ids = self._library.persist_entries(entries, problem=problem.name, gpu=self.config.gpu)
            self.emit("library_updated", count=len(ids))
        except Exception as exc:  # noqa: BLE001 — never fail a run on bookkeeping
            self.emit("library_error", error=str(exc))

    # ----------------------------------------------------------------- #
    # Evaluation (ground truth)
    # ----------------------------------------------------------------- #
    def _evaluate(
        self, record: CandidateRecord, problem: Problem, reference_ns: float | None
    ) -> EvalOutcome:
        if not self.config.hardware.can_run_locally:
            claimed = bool(record.candidate and record.candidate.claimed_passing)
            return EvalOutcome(
                passed=claimed, source="agent-claim",
                detail="leaderboard-only: trusting agent self-report (no local GPU)",
            )

        submission = record.workspace / problem.submission_filename
        if not submission.exists():
            return EvalOutcome(passed=False, error="no submission file written")

        code = submission.read_text()
        try:
            evalproto.stage_workspace(record.workspace, problem, code)
            test = evalproto.run_eval(record.workspace, "test", problem)
        except evalproto.LocalEvalError as exc:
            return EvalOutcome(passed=False, error=f"eval error: {exc}")
        if not test.passed:
            return EvalOutcome(passed=False, detail=test.error or "failed correctness", error=test.error)

        bench = evalproto.run_eval(record.workspace, "benchmark", problem)
        speedup = compute_speedup(reference_ns, bench.geomean_ns)
        return EvalOutcome(
            passed=bench.passed,
            geomean_ns=bench.geomean_ns,
            speedup=speedup,
            source="local",
            detail=f"geomean={bench.geomean_ns:.1f}ns" if bench.geomean_ns else "",
            error=bench.error,
        )

    # ----------------------------------------------------------------- #
    # Workspace / context helpers
    # ----------------------------------------------------------------- #
    def _workspace(self, name: str) -> Path:
        return self._runs_dir / self._run_id / name

    def _context(self, problem: Problem, workspace: Path) -> RunContext:
        import sys

        return RunContext(
            problem=problem,
            workspace=workspace,
            gpu=self.config.gpu,
            hardware=self.config.hardware,
            python_bin=sys.executable,
        )

    def _seed_workspace(self, dst: Path, src: Path, problem: Problem) -> None:
        """Copy the current best submission into a fresh rewrite workspace."""
        dst.mkdir(parents=True, exist_ok=True)
        src_sub = src / problem.submission_filename
        if src_sub.exists():
            (dst / problem.submission_filename).write_text(src_sub.read_text())

    def _hardware_notes(self) -> str:
        if self.config.hardware is HardwareProfile.BLACKWELL:
            return (
                "Target is an NVIDIA Blackwell GPU (GB10 class). Reason about "
                "Blackwell's memory hierarchy, tensor cores (incl. NVFP4), and "
                "bandwidth — not generic CUDA advice."
            )
        if self.config.hardware is HardwareProfile.NVIDIA:
            return "Target is an NVIDIA (non-Blackwell) GPU; use standard CUDA optimization."
        return ""
