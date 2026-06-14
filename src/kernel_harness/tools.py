"""Local tool layer — the hands the subagents act through.

These are in-process MCP tools (Claude Agent SDK ``@tool`` + ``create_sdk_mcp_server``)
bound to a :class:`RunContext`. They give the kernel-writer and profiler subagents
the ability to:

* ``save_submission``   — write a candidate kernel to the workspace
* ``run_local``         — test/benchmark it on the local GPU via the eval harness
* ``profile_kernel``    — run Nsight Compute (ncu) on it
* ``popcorn_submit``    — submit to the GPU MODE leaderboard via popcorn-cli

The submission file on disk is the single source of truth — ``run_local`` and
``profile_kernel`` read whatever is there (whether written by ``save_submission``
or the agent's built-in Write tool).

Hardware gating: ``run_local`` / ``profile_kernel`` are no-ops with an explanatory
message on the ``leaderboard-only`` profile (CPU / non-NVIDIA), where the loop
runs purely off popcorn output. See CONTEXT.md → "Hardware Profiles".

The subprocess-driven tools (run_local, profile, popcorn) can only be fully
exercised on a real GPU box; their command construction is factored into small
helpers so it's inspectable and unit-testable offline.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import evalproto
from .config import HardwareProfile
from .problem import Problem

MCP_SERVER_NAME = "kernel_tools"


@dataclass
class RunContext:
    """Per-run state shared by every tool."""

    problem: Problem
    workspace: Path
    gpu: str
    hardware: HardwareProfile
    python_bin: str = sys.executable
    popcorn_bin: str = "popcorn"
    ncu_bin: str = "ncu"
    _support_staged: bool = field(default=False, repr=False)

    @property
    def submission_path(self) -> Path:
        return self.workspace / self.problem.submission_filename

    def ensure_support_staged(self) -> None:
        """Write the eval harness's support + test files once (idempotent)."""
        if self._support_staged:
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        for name, content in self.problem.stage_files.items():
            (self.workspace / name).write_text(content)
        (self.workspace / "tests.txt").write_text(
            evalproto.render_test_file(self.problem.tests)
        )
        (self.workspace / "benchmarks.txt").write_text(
            evalproto.render_test_file(self.problem.benchmarks)
        )
        self._support_staged = True


# --------------------------------------------------------------------------- #
# Result formatting helpers
# --------------------------------------------------------------------------- #
def _text(s: str, *, is_error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"content": [{"type": "text", "text": s}]}
    if is_error:
        out["is_error"] = True
    return out


def _summarize_eval(result: evalproto.EvalResult) -> str:
    """Render an EvalResult into a compact, agent-readable summary."""
    head = f"mode={result.mode} check={result.check} passed={result.passed} exit={result.exit_code}"
    lines = [head]
    if result.is_timing:
        lines.append(f"geomean={result.geomean_ns:.1f} ns over {len(result.benchmark_means_ns)} benchmarks")
        for c in result.cases:
            mean = c.get("mean")
            if isinstance(mean, float):
                lines.append(f"  bench[{c['index']}] {c.get('spec','')}: mean={mean:.1f}ns best={c.get('best','?')}")
    else:
        for c in result.cases:
            lines.append(f"  test[{c['index']}] {c.get('spec','')}: {c.get('status','?')}")
    if result.error:
        lines.append(f"error: {result.error}")
    if result.stderr.strip():
        lines.append(f"stderr (tail):\n{result.stderr.strip()[-1500:]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command builders (pure — unit-tested)
# --------------------------------------------------------------------------- #
def build_popcorn_cmd(ctx: RunContext, mode: str) -> list[str]:
    """Construct the popcorn-cli submission command."""
    return [
        ctx.popcorn_bin,
        "submit",
        "--leaderboard",
        ctx.problem.name,
        "--gpu",
        ctx.gpu,
        "--mode",
        mode,
        str(ctx.submission_path),
    ]


def build_ncu_cmd(ctx: RunContext, *, sections: str | None = None) -> list[str]:
    """Construct an Nsight Compute command that profiles the benchmark run.

    Runs ncu around the eval harness in ``benchmark`` mode so the real kernels
    execute. ``--set`` selects the metric sections; default is a basic set to
    keep runtime bounded.
    """
    cmd = [
        ctx.ncu_bin,
        "--target-processes",
        "all",
        "--set",
        sections or "basic",
        ctx.python_bin,
        ctx.problem.entry_point,
        "benchmark",
        "benchmarks.txt",
    ]
    return cmd


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #
def build_tool_server(ctx: RunContext) -> tuple[dict[str, Any], list[str]]:
    """Build the SDK MCP server for ``ctx`` and the allowed-tool names.

    Returns ``(mcp_servers_dict, allowed_tool_names)`` ready to hand to
    ``AgentRunner(mcp_servers=...)`` and ``run_subagent(allowed_tools=...)``.
    """

    @tool("save_submission", "Write the candidate kernel to the workspace submission file. "
          "Validates that the code compiles. Returns the path.", {"code": str})
    async def save_submission(args: dict) -> dict:
        code = args["code"]
        try:
            compile(code, ctx.problem.submission_filename, "exec")
        except SyntaxError as exc:
            return _text(f"submission has a syntax error: {exc}", is_error=True)
        ctx.submission_path.write_text(code)
        return _text(f"saved {len(code)} bytes to {ctx.submission_path}")

    @tool("run_local", "Test or benchmark the current submission on the LOCAL GPU via the "
          "GPU MODE eval harness. mode is 'test' (correctness) or 'benchmark' (timing). "
          "Returns pass/fail and per-case timings.", {"mode": str})
    async def run_local(args: dict) -> dict:
        if not ctx.hardware.can_run_locally:
            return _text(
                "local execution unavailable on this hardware profile "
                f"({ctx.hardware.value}); use popcorn_submit instead.",
                is_error=True,
            )
        mode = args.get("mode", "test")
        if mode not in ("test", "benchmark"):
            return _text(f"run_local mode must be 'test' or 'benchmark', got {mode!r}", is_error=True)
        if not ctx.submission_path.exists():
            return _text("no submission saved yet; call save_submission first", is_error=True)

        ctx.ensure_support_staged()
        try:
            result = await asyncio.to_thread(
                evalproto.run_eval, ctx.workspace, mode, ctx.problem,
                python_bin=ctx.python_bin,
            )
        except evalproto.LocalEvalError as exc:
            return _text(f"local eval error: {exc}", is_error=True)
        return _text(_summarize_eval(result), is_error=not result.passed)

    @tool("profile_kernel", "Profile the current submission with Nsight Compute (ncu) on the "
          "local GPU and return the report text. Optional 'sections' selects the ncu --set "
          "(e.g. 'basic', 'full', 'roofline').", {"sections": str})
    async def profile_kernel(args: dict) -> dict:
        if not ctx.hardware.can_profile:
            return _text(
                f"profiling unavailable on hardware profile {ctx.hardware.value}.",
                is_error=True,
            )
        if not ctx.submission_path.exists():
            return _text("no submission saved yet; call save_submission first", is_error=True)
        ctx.ensure_support_staged()
        cmd = build_ncu_cmd(ctx, sections=args.get("sections") or None)
        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd,
                cwd=str(ctx.workspace), capture_output=True, text=True,
                timeout=ctx.problem.timeouts.get("benchmark_timeout", 600) * 3,
            )
        except FileNotFoundError:
            return _text(f"ncu not found ({ctx.ncu_bin}); is Nsight Compute installed?", is_error=True)
        except subprocess.TimeoutExpired:
            return _text("ncu profiling timed out", is_error=True)
        report = proc.stdout or proc.stderr
        return _text(f"$ {shlex.join(cmd)}\n\n{report[-12000:]}", is_error=proc.returncode != 0)

    @tool("popcorn_submit", "Submit the current submission to the GPU MODE leaderboard via "
          "popcorn-cli. mode is 'test' (remote correctness), 'benchmark' (remote timing, "
          "unranked), or 'leaderboard' (ranked). Returns popcorn's output.", {"mode": str})
    async def popcorn_submit(args: dict) -> dict:
        mode = args.get("mode", "test")
        if mode not in ("test", "benchmark", "leaderboard", "profile"):
            return _text(f"popcorn mode invalid: {mode!r}", is_error=True)
        if not ctx.submission_path.exists():
            return _text("no submission saved yet; call save_submission first", is_error=True)
        cmd = build_popcorn_cmd(ctx, mode)
        try:
            proc = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=1800,
            )
        except FileNotFoundError:
            return _text(f"popcorn not found ({ctx.popcorn_bin}); is popcorn-cli installed/authed?", is_error=True)
        except subprocess.TimeoutExpired:
            return _text("popcorn submission timed out", is_error=True)
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return _text(f"$ {shlex.join(cmd)}\n\n{out.strip()[-12000:]}", is_error=proc.returncode != 0)

    tools = [save_submission, run_local, profile_kernel, popcorn_submit]
    server = create_sdk_mcp_server(MCP_SERVER_NAME, tools=tools)
    allowed = [f"mcp__{MCP_SERVER_NAME}__{t.name}" for t in tools]
    return {MCP_SERVER_NAME: server}, allowed
