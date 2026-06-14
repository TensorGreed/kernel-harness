"""Offline tests for the eval protocol + tool layer.

Covers the pure parsers/renderers, the command builders, an end-to-end
``run_eval`` against a stub ``eval.py`` that mimics the POPCORN_FD protocol
(no GPU, no torch), and the MCP tool-server assembly.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_evalproto
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from kernel_harness import evalproto
from kernel_harness.config import HardwareProfile
from kernel_harness.problem import Problem
from kernel_harness.tools import build_ncu_cmd, build_popcorn_cmd, build_tool_server, RunContext


def test_render_test_file() -> None:
    out = evalproto.render_test_file([{"size": 127, "seed": 4242}, {"size": 128, "seed": 5236}])
    assert out == "size: 127; seed: 4242\nsize: 128; seed: 5236\n", repr(out)
    assert evalproto.render_test_file([]) == ""
    print("ok  test_render_test_file")


def test_parse_test_mode() -> None:
    text = "test-count: 2\ntest.0.spec: size: 127\ntest.0.status: pass\ntest.1.status: pass\ncheck: pass\n"
    r = evalproto.parse_eval_output("test", text, exit_code=0)
    assert r.passed
    assert r.check == "pass"
    assert len(r.cases) == 2
    assert not r.is_timing

    fail = "test.0.status: fail\ntest.0.error: mismatch at index 3\ncheck: fail\n"
    rf = evalproto.parse_eval_output("test", fail, exit_code=112)
    assert not rf.passed
    assert rf.error == "mismatch at index 3", rf.error
    print("ok  test_parse_test_mode")


def test_parse_benchmark_mode() -> None:
    text = (
        "benchmark-count: 2\n"
        "benchmark.0.spec: size: 1024\nbenchmark.0.runs: 100\nbenchmark.0.mean: 1000.0\nbenchmark.0.best: 950.0\n"
        "benchmark.1.spec: size: 2048\nbenchmark.1.runs: 100\nbenchmark.1.mean: 4000.0\nbenchmark.1.best: 3900.0\n"
        "check: pass\n"
    )
    r = evalproto.parse_eval_output("benchmark", text, exit_code=0)
    assert r.passed
    assert r.is_timing
    assert r.benchmark_means_ns == [1000.0, 4000.0], r.benchmark_means_ns
    # geomean(1000, 4000) = 2000
    assert abs(r.geomean_ns - 2000.0) < 1e-6, r.geomean_ns
    print("ok  test_parse_benchmark_mode")


def _stub_problem(workspace_marker: str = "") -> Problem:
    """A Problem whose eval.py is a torch-free stub honoring the FD protocol."""
    stub_eval = r'''
import os, sys
fd = int(os.environ["POPCORN_FD"])
mode, testfile = sys.argv[1], sys.argv[2]
lines = open(testfile).read().splitlines()
out = os.fdopen(fd, "w")
if mode == "test":
    out.write("test-count: %d\n" % len(lines))
    for i, spec in enumerate(lines):
        out.write("test.%d.status: pass\n" % i)
    out.write("check: pass\n")
elif mode == "benchmark":
    out.write("benchmark-count: %d\n" % len(lines))
    for i, spec in enumerate(lines):
        out.write("benchmark.%d.spec: %s\n" % (i, spec))
        out.write("benchmark.%d.runs: 100\n" % i)
        out.write("benchmark.%d.mean: %d\n" % (i, 1000 * (i + 1)))
    out.write("check: pass\n")
out.flush(); out.close()
sys.exit(0)
'''
    return Problem(
        name="stub_v2",
        problem_set="stub",
        directory="stub/stub_py",
        gpus=["B200"],
        spec_files={},
        entry_point="eval.py",
        submission_filename="submission.py",
        stage_files={"eval.py": stub_eval, "utils.py": "# noop\n"},
        tests=[{"size": 127, "seed": 1}, {"size": 128, "seed": 2}],
        benchmarks=[{"size": 1024, "seed": 9}, {"size": 2048, "seed": 10}],
        timeouts={"test_timeout": 60, "benchmark_timeout": 60},
    )


def test_run_eval_end_to_end() -> None:
    problem = _stub_problem()
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        evalproto.stage_workspace(ws, problem, "def custom_kernel(x): return x\n")
        assert (ws / "eval.py").exists()
        assert (ws / "submission.py").exists()
        assert (ws / "tests.txt").read_text().startswith("size: 127")

        # test mode
        rt = evalproto.run_eval(ws, "test", problem, python_bin=sys.executable)
        assert rt.passed, (rt.exit_code, rt.stderr)
        assert len(rt.cases) == 2

        # benchmark mode -> timing + geomean(1000, 2000)
        rb = evalproto.run_eval(ws, "benchmark", problem, python_bin=sys.executable)
        assert rb.passed, (rb.exit_code, rb.stderr)
        assert rb.benchmark_means_ns == [1000.0, 2000.0], rb.benchmark_means_ns
        assert abs(rb.geomean_ns - (1000 * 2000) ** 0.5) < 1e-6
    print("ok  test_run_eval_end_to_end")


def _ctx(tmp: Path) -> RunContext:
    return RunContext(
        problem=_stub_problem(),
        workspace=tmp,
        gpu="B200",
        hardware=HardwareProfile.BLACKWELL,
        python_bin=sys.executable,
    )


def test_command_builders() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctx = _ctx(Path(td))
        pc = build_popcorn_cmd(ctx, "leaderboard")
        assert pc[:6] == ["popcorn", "submit", "--leaderboard", "stub_v2", "--gpu", "B200"], pc
        assert pc[-3:] == ["--mode", "leaderboard", str(ctx.submission_path)], pc

        ncu = build_ncu_cmd(ctx, sections="roofline")
        assert ncu[0] == "ncu"
        assert "--set" in ncu and ncu[ncu.index("--set") + 1] == "roofline"
        assert ncu[-3:] == ["benchmark", "benchmarks.txt"] or ncu[-2:] == ["benchmark", "benchmarks.txt"], ncu
    print("ok  test_command_builders")


def test_build_tool_server() -> None:
    with tempfile.TemporaryDirectory() as td:
        ctx = _ctx(Path(td))
        servers, allowed = build_tool_server(ctx)
        assert "kernel_tools" in servers
        assert servers["kernel_tools"]["type"] == "sdk"
        assert set(allowed) == {
            "mcp__kernel_tools__save_submission",
            "mcp__kernel_tools__run_local",
            "mcp__kernel_tools__profile_kernel",
            "mcp__kernel_tools__popcorn_submit",
        }, allowed
    print("ok  test_build_tool_server")


if __name__ == "__main__":
    test_render_test_file()
    test_parse_test_mode()
    test_parse_benchmark_mode()
    test_run_eval_end_to_end()
    test_command_builders()
    test_build_tool_server()
    print("\nall evalproto + tools checks passed")
