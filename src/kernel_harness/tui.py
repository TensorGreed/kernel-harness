"""Attachable Textual dashboard for a run.

The harness runs autonomously; this is the optional cockpit. It drives an
``Orchestrator`` in a worker, streams its ``on_event`` feed into a live event log
and a candidate table, and lets the user intervene:

* ``s`` — force-stop after the current iteration (``Orchestrator.request_stop``)
* ``q`` — quit

The orchestrator is fully decoupled — the app only needs an object exposing
``set_on_event(cb)``, ``request_stop()``, and an async ``run()`` returning a
``RunReport``. The event→UI mapping (``candidate_row`` / ``format_log_line``) is
pure and unit-tested; the app itself is smoke-tested with Textual's ``run_test``.
See CONTEXT.md → "Interaction Model".
"""

from __future__ import annotations

from typing import Any, Protocol

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, RichLog, Static


class OrchestratorLike(Protocol):
    def set_on_event(self, cb) -> None: ...
    def request_stop(self) -> None: ...
    async def run(self) -> Any: ...


# --------------------------------------------------------------------------- #
# Pure event → UI mapping
# --------------------------------------------------------------------------- #
_COLUMNS = ("candidate", "approach", "status", "speedup")


def candidate_row(payload: dict) -> tuple[str, str, str, str]:
    """Map a ``candidate_evaluated`` payload to a table row."""
    spd = payload.get("speedup")
    spd_s = f"{spd:.2f}x" if isinstance(spd, (int, float)) and spd else "—"
    status = "✓ pass" if payload.get("passed") else "✗ fail"
    return (str(payload.get("id", "?")), str(payload.get("approach", "")), status, spd_s)


def format_log_line(name: str, payload: dict) -> str | None:
    """Render an event as a log line, or None to skip it."""
    match name:
        case "run_start":
            return f"▶ optimizing {payload.get('problem')} on {payload.get('gpu')}"
        case "brief":
            return f"understood: {payload.get('summary')}"
        case "baseline":
            ref = payload.get("reference_ns")
            return f"reference baseline: {ref:.1f} ns" if ref else "reference baseline: unavailable"
        case "iteration_start":
            return f"── iteration {payload.get('index')}"
        case "candidate_evaluated":
            return None  # shown in the table
        case "candidate_failed":
            return f"✗ {payload.get('approach')} failed: {payload.get('error')}"
        case "research":
            return f"research triggered: {payload.get('trigger')}"
        case "decision":
            return f"strategy: {payload.get('action')} — {payload.get('focus', '')}"
        case "notice":
            return f"! {payload.get('message')}"
        case "library_updated":
            return f"library: persisted {payload.get('count')} lessons"
        case "submitting":
            return f"submitting: {payload.get('cmd')}"
        case "submitted":
            return f"submitted: ok={payload.get('ok')}"
        case "run_end":
            return f"■ done — {payload.get('reason')}"
        case _ if name.endswith("_error"):
            return f"[error] {name}: {payload}"
        case _:
            return None


def format_report(report: Any) -> str:
    """One-line best-result summary for the status bar."""
    if report is None:
        return "no report"
    best = getattr(report, "best", None)
    if best and best.outcome.passed:
        spd = best.outcome.speedup
        spd_s = f"{spd:.2f}x reference" if spd else "n/a"
        return f"best: {best.id} ({best.approach}) — {spd_s}  |  stopped: {report.stopped_reason}"
    return f"best: none  |  stopped: {report.stopped_reason}"


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
class HarnessApp(App):
    """Live dashboard around an Orchestrator run."""

    BINDINGS = [
        ("s", "stop", "Force stop"),
        ("q", "quit", "Quit"),
    ]
    CSS = """
    #candidates { height: 1fr; border: round $primary; }
    #log { width: 1fr; border: round $secondary; }
    #status { height: 1; background: $boost; padding: 0 1; }
    """

    def __init__(self, orchestrator: OrchestratorLike, *, problem: str = "", gpu: str = "") -> None:
        super().__init__()
        self._orch = orchestrator
        self._problem = problem
        self._gpu = gpu
        self._seen_rows: set[str] = set()
        self._col_keys: list = []
        self._done = False
        self._status_text = "starting…"  # mirror of the status bar, for tests

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield DataTable(id="candidates")
            yield RichLog(id="log", highlight=False, markup=False, wrap=True)
        yield Static("starting…", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "kernel-harness"
        self.sub_title = f"{self._problem} · {self._gpu}"
        table = self.query_one("#candidates", DataTable)
        self._col_keys = list(table.add_columns(*_COLUMNS))
        self._orch.set_on_event(self._on_event)
        self.run_worker(self._drive(), exclusive=True)

    async def _drive(self) -> None:
        try:
            report = await self._orch.run()
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the UI
            self._log_line(f"[run error] {exc}")
            self._set_status(f"run failed: {exc}")
            self._done = True
            return
        self._set_status(format_report(report))
        self._log_line("run complete — press q to quit")
        self._done = True

    # The orchestrator emits synchronously from this same event loop (the worker),
    # so apply directly — deferring via call_later would let the worker's final
    # status write land before these events drain, reordering the UI.
    def _on_event(self, name: str, payload: dict) -> None:
        try:
            self.apply_event(name, payload)
        except Exception:  # noqa: BLE001 — a UI hiccup must not break the run
            pass

    def apply_event(self, name: str, payload: dict) -> None:
        """Update the UI for one event (also called directly in tests)."""
        if name == "candidate_evaluated":
            self._upsert_candidate(payload)
        line = format_log_line(name, payload)
        if line:
            self._log_line(line)
        if name == "run_start":
            self.sub_title = f"{payload.get('problem')} · {payload.get('gpu')}"
        elif name == "iteration_start":
            self._set_status(f"iteration {payload.get('index')} running…")

    # ----------------------------------------------------------------- #
    def _upsert_candidate(self, payload: dict) -> None:
        table = self.query_one("#candidates", DataTable)
        cid, approach, status, spd = candidate_row(payload)
        if cid in self._seen_rows:
            for col_key, value in zip(self._col_keys, (cid, approach, status, spd)):
                table.update_cell(cid, col_key, value)
        else:
            table.add_row(cid, approach, status, spd, key=cid)
            self._seen_rows.add(cid)

    def _log_line(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.query_one("#status", Static).update(text)

    # ----------------------------------------------------------------- #
    def action_stop(self) -> None:
        self._orch.request_stop()
        self._log_line("! stop requested — finishing current iteration")
        self._set_status("stopping after current iteration…")


def run_tui(orchestrator: OrchestratorLike, *, problem: str = "", gpu: str = "") -> None:
    """Launch the dashboard (blocks until quit)."""
    HarnessApp(orchestrator, problem=problem, gpu=gpu).run()
