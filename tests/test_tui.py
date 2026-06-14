"""Offline tests for the TUI.

Pure event→UI mappers are tested directly; the app is smoke-tested with
Textual's ``run_test`` against a fake orchestrator (no model, no GPU). Verifies
the candidate table fills, the event log writes, force-stop calls through, and a
final report lands in the status bar.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_tui
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kernel_harness.tui import HarnessApp, candidate_row, format_log_line, format_report


def test_candidate_row() -> None:
    assert candidate_row({"id": "i1-c0-triton", "approach": "triton", "passed": True, "speedup": 1.8}) == (
        "i1-c0-triton", "triton", "✓ pass", "1.80x"
    )
    miss = candidate_row({"id": "x", "approach": "cuda", "passed": False, "speedup": None})
    assert miss[2] == "✗ fail" and miss[3] == "—"
    print("ok  test_candidate_row")


def test_format_log_line() -> None:
    assert "optimizing" in format_log_line("run_start", {"problem": "vectoradd_v2", "gpu": "B200"})
    assert format_log_line("candidate_evaluated", {}) is None  # table only
    assert "iteration 2" in format_log_line("iteration_start", {"index": 2})
    assert "pay-as-you-go" in format_log_line("notice", {"message": "pay-as-you-go"})
    assert "unavailable" in format_log_line("baseline", {"reference_ns": None})
    assert format_log_line("baseline_error", {"error": "x"}).startswith("[error]")
    assert format_log_line("totally_unknown", {}) is None
    print("ok  test_format_log_line")


@dataclass
class _Outcome:
    passed: bool = True
    speedup: float | None = 2.0


@dataclass
class _Best:
    id: str = "i1-c0-triton"
    approach: str = "triton"
    outcome: _Outcome = None  # type: ignore[assignment]


@dataclass
class _Report:
    best: object
    stopped_reason: str = "iteration budget reached (2)"


def test_format_report() -> None:
    rep = _Report(best=_Best(outcome=_Outcome()))
    assert "2.00x reference" in format_report(rep)
    assert "best: none" in format_report(_Report(best=None))
    assert format_report(None) == "no report"
    print("ok  test_format_report")


class FakeOrchestrator:
    """Emits a scripted event stream, then returns a report."""

    def __init__(self) -> None:
        self._cb = lambda name, payload: None
        self.stop_called = False

    def set_on_event(self, cb) -> None:
        self._cb = cb

    def request_stop(self) -> None:
        self.stop_called = True

    async def run(self):
        self._cb("run_start", {"problem": "vectoradd_v2", "gpu": "B200"})
        self._cb("iteration_start", {"index": 1})
        self._cb("candidate_evaluated", {"id": "i1-c0-triton", "approach": "triton", "passed": True, "speedup": 1.8})
        self._cb("candidate_evaluated", {"id": "i1-c1-cuda", "approach": "cuda-inline", "passed": False, "speedup": None})
        self._cb("run_end", {"reason": "iteration budget reached (1)"})
        return _Report(best=_Best(outcome=_Outcome(speedup=1.8)))


def test_app_smoke() -> None:
    from textual.widgets import DataTable, Static

    async def scenario() -> None:
        orch = FakeOrchestrator()
        app = HarnessApp(orch, problem="vectoradd_v2", gpu="B200")
        async with app.run_test() as pilot:
            # Let on_mount start the worker and the scripted run drain.
            await pilot.pause()
            await pilot.pause()

            table = app.query_one("#candidates", DataTable)
            assert table.row_count == 2, table.row_count

            # The scripted run has finished -> final report in the status mirror.
            assert "best" in app._status_text.lower(), app._status_text

            # Force-stop binding routes through to the orchestrator.
            await pilot.press("s")
            assert orch.stop_called
            assert "stopping" in app._status_text.lower()

    asyncio.run(scenario())
    print("ok  test_app_smoke")


if __name__ == "__main__":
    test_candidate_row()
    test_format_log_line()
    test_format_report()
    test_app_smoke()
    print("\nall tui checks passed")
