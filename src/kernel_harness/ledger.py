"""Persistent on-disk experiment ledger.

Every run writes a durable record under ``runs/<run_id>/``:

* ``summary.md`` — the master log: a header (problem, brief, GPU, reference
  baseline, workload structure) followed by one table row per evaluated
  candidate, plus per-iteration profiler/decision notes appended chronologically.
* ``<candidate>/result.md`` — per-candidate detail (approach, pass/fail, timing,
  speedup, model/backend/cost, the writer's summary, any error) alongside the
  candidate's ``submission.py`` snapshot the harness already wrote there.

This is the shared, inspectable memory of a run. ``reflect`` and
``update_library`` read ``render()`` (the on-disk summary) instead of in-memory
state, so the history is durable, human-readable, resumable after a crash, and
— crucially — the substrate a future clean-context "research" agent reads via
file tools rather than receiving an ever-growing dump in its prompt.

Writes are best-effort: a disk hiccup must never kill an expensive run. See
CONTEXT.md → "Post-design additions".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import CandidateRecord, IterationRecord
    from .subagents import ProblemBrief, WorkloadProfile

_SUMMARY_COLUMNS = "| Iter | Candidate | Approach | Pass | geomean(ns) | Speedup | Note |"
_SUMMARY_DIVIDER = "|---|---|---|---|---|---|---|"


def _fmt_ns(ns: float | None) -> str:
    return f"{ns:.1f}" if ns is not None else "—"


def _fmt_speedup(s: float | None) -> str:
    return f"{s:.2f}x" if s else "—"


class Ledger:
    """The on-disk record for one optimization run."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.summary_path = root / "summary.md"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # ----------------------------------------------------------------- #
    def write_header(
        self,
        *,
        problem: str,
        brief: "ProblemBrief | None",
        gpu: str,
        reference_ns: float | None,
        workload: "WorkloadProfile | None",
    ) -> None:
        lines = [
            f"# Run — {problem}",
            "",
            f"- GPU: {gpu}",
            f"- reference baseline (ns): {_fmt_ns(reference_ns)}",
        ]
        if brief is not None:
            lines += [
                f"- summary: {brief.summary}",
                f"- dtype: {brief.dtype}",
                f"- kernel signature: {brief.kernel_signature}",
            ]
            if brief.optimization_targets:
                lines.append(f"- optimization targets: {', '.join(brief.optimization_targets)}")
        if workload is not None and (workload.shortcuts or workload.exploitable_structure):
            lines.append("")
            lines.append("## Workload structure")
            if workload.summary:
                lines.append(f"- {workload.summary}")
            for s in workload.exploitable_structure:
                lines.append(f"- structure: {s}")
            for s in workload.shortcuts:
                lines.append(f"- shortcut: {s}")
        lines += ["", "## Candidates", "", _SUMMARY_COLUMNS, _SUMMARY_DIVIDER, ""]
        self._write(self.summary_path, "\n".join(lines) + "\n")

    def record_candidate(
        self, *, iteration: int, record: "CandidateRecord", is_best: bool, note: str = ""
    ) -> None:
        o = record.outcome
        cand = record.candidate
        # Per-candidate detail file next to its submission snapshot.
        detail = [
            f"# {record.id} — iteration {iteration}",
            "",
            f"- approach: {record.approach}",
            f"- pass: {o.passed}",
            f"- geomean (ns): {_fmt_ns(o.geomean_ns)}",
            f"- speedup: {_fmt_speedup(o.speedup)}",
            f"- eval source: {o.source}",
        ]
        if cand is not None:
            detail.append(f"- writer summary: {cand.summary}")
            if cand.raw is not None:
                r = cand.raw
                detail.append(f"- model: {r.model} ({r.backend})")
                if r.cost_usd is not None:
                    detail.append(f"- cost (usd): {r.cost_usd}")
        if o.error:
            detail.append(f"- error: {o.error}")
        try:
            record.workspace.mkdir(parents=True, exist_ok=True)
            self._write(record.workspace / "result.md", "\n".join(detail) + "\n")
        except OSError:
            pass

        status = "pass" if o.passed else "FAIL"
        marker = "**best**" if is_best else (note or "")
        row = (
            f"| {iteration} | {record.id} | {record.approach} | {status} | "
            f"{_fmt_ns(o.geomean_ns)} | {_fmt_speedup(o.speedup)} | {marker} |"
        )
        self._append(row)

    def record_iteration_note(self, iteration: int, it_record: "IterationRecord") -> None:
        notes = []
        if it_record.profiler is not None and it_record.profiler.bottleneck:
            notes.append(f"profiler: {it_record.profiler.bottleneck}")
        if getattr(it_record, "trigger", None):
            notes.append(f"STUCK ({it_record.trigger})")
        if getattr(it_record, "research", None) is not None:
            r = it_record.research
            notes.append(f"research: {r.strategy} — {r.diagnosis}")
        if it_record.decision is not None:
            d = it_record.decision
            extra = f" — {d.focus}" if d.focus else ""
            notes.append(f"decision: {d.action}{extra}")
        if notes:
            self._append(f"\n_iteration {iteration}: " + "; ".join(notes) + "_\n")

    def render(self) -> str:
        """Return the on-disk summary — the history string for reflect/update_library."""
        try:
            return self.summary_path.read_text()
        except OSError:
            return ""

    # ----------------------------------------------------------------- #
    def _append(self, line: str) -> None:
        try:
            with self.summary_path.open("a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    @staticmethod
    def _write(path: Path, content: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        except OSError:
            pass
