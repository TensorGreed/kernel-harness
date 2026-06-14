"""The GPU MODE local-eval protocol — stage, run, parse.

``popcorn submit`` runs on the *remote* leaderboard GPU. To test/benchmark a
candidate on the local GB10 we reproduce GPU MODE's own eval harness
(``eval.py``) exactly:

* **Stage** a workspace with the problem's ``stage_files`` (eval.py, utils.py,
  task.py, reference.py), the candidate ``submission.py``, and a test-spec file
  generated from the problem's ``tests``/``benchmarks`` lists.
* **Run** ``python eval.py <mode> <testfile>`` with ``POPCORN_FD`` pointing at a
  file descriptor the harness writes its structured results to (it does *not*
  use stdout). Modes: ``test`` (correctness), ``benchmark`` (timing),
  ``leaderboard`` (ranked timing), ``profile`` (torch profiler).
* **Parse** the ``key: value`` lines it emits — ``check: pass|fail``,
  ``test.<i>.status``, ``benchmark.<i>.{runs,mean,std,err,best,worst}`` (ns).

The pure pieces (``render_test_file``, ``parse_eval_output``) are unit-tested
offline; ``run_eval`` is validated against a stub eval script (no GPU needed).
A real kernel run requires the GB10. See CONTEXT.md → "Profiling Strategy".
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .problem import Problem

# Eval exit codes (from eval.py).
EXIT_PASS = 0
EXIT_FAIL = 112
EXIT_NO_FD = 111
EXIT_BAD_TESTFILE = 113

_BENCH_STAT_FIELDS = ("runs", "mean", "std", "err", "best", "worst")
_CASE_KEY_RE = re.compile(r"^(test|benchmark)\.(\d+)\.(\w+)$")


class LocalEvalError(RuntimeError):
    """Raised when the eval harness can't be run (staging/plumbing failure)."""


# --------------------------------------------------------------------------- #
# Test-spec file generation (pure)
# --------------------------------------------------------------------------- #
def render_test_file(cases: list[dict]) -> str:
    """Render case dicts into the eval harness's test-spec format.

    Each case like ``{"size": 127, "seed": 4242}`` becomes a line
    ``size: 127; seed: 4242``. Key order is preserved.
    """
    lines = []
    for case in cases:
        parts = [f"{k}: {v}" for k, v in case.items()]
        lines.append("; ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


# --------------------------------------------------------------------------- #
# Output parsing (pure)
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    """Structured outcome of one eval-harness run."""

    mode: str
    exit_code: int
    passed: bool
    check: str | None = None
    cases: list[dict] = field(default_factory=list)       # per-index merged fields
    benchmark_means_ns: list[float] = field(default_factory=list)
    geomean_ns: float | None = None                       # single timing score
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def is_timing(self) -> bool:
        return self.geomean_ns is not None


def parse_eval_output(
    mode: str,
    fd_text: str,
    *,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
) -> EvalResult:
    """Parse the harness's ``key: value`` output into an ``EvalResult``."""
    raw: dict[str, str] = {}
    cases_by_idx: dict[int, dict] = {}

    for line in fd_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        raw[key] = value
        if (m := _CASE_KEY_RE.match(key)):
            idx, fieldname = int(m.group(2)), m.group(3)
            case = cases_by_idx.setdefault(idx, {"index": idx})
            # Coerce the numeric stat fields to float.
            if fieldname in _BENCH_STAT_FIELDS:
                try:
                    value = float(value)  # type: ignore[assignment]
                except ValueError:
                    pass
            case[fieldname] = value

    cases = [cases_by_idx[i] for i in sorted(cases_by_idx)]
    check = raw.get("check")
    passed = check == "pass" and exit_code == EXIT_PASS

    # Collect per-benchmark mean durations (ns) and reduce to a single score.
    means = [c["mean"] for c in cases if isinstance(c.get("mean"), float)]
    geomean = _geomean(means) if means else None

    # First error we can find, for surfacing to the loop.
    error = None
    if not passed:
        for c in cases:
            if c.get("status") == "fail" and c.get("error"):
                error = str(c["error"])
                break
        if error is None:
            error = stderr.strip() or f"eval failed (exit {exit_code}, check={check})"

    return EvalResult(
        mode=mode,
        exit_code=exit_code,
        passed=passed,
        check=check,
        cases=cases,
        benchmark_means_ns=means,
        geomean_ns=geomean,
        stdout=stdout,
        stderr=stderr,
        error=error,
        raw=raw,
    )


def _geomean(values: list[float]) -> float | None:
    """Geometric mean of positive durations (GPU MODE's score reduction)."""
    positive = [v for v in values if v > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


# --------------------------------------------------------------------------- #
# Staging (I/O)
# --------------------------------------------------------------------------- #
def stage_workspace(
    workspace: Path,
    problem: Problem,
    submission_code: str,
) -> dict[str, Path]:
    """Write everything the eval harness needs into ``workspace``.

    Returns a map of logical name -> path for the files the runner cares about
    (``tests``, ``benchmarks``, and the submission).
    """
    workspace.mkdir(parents=True, exist_ok=True)

    # Harness support files (eval.py, utils.py, task.py, reference.py).
    for name, content in problem.stage_files.items():
        _write(workspace / name, content)

    # The candidate.
    submission_path = workspace / problem.submission_filename
    _write(submission_path, submission_code)

    # Test-spec files materialized from task.yml.
    tests_path = workspace / "tests.txt"
    benchmarks_path = workspace / "benchmarks.txt"
    _write(tests_path, render_test_file(problem.tests))
    _write(benchmarks_path, render_test_file(problem.benchmarks))

    return {
        "submission": submission_path,
        "tests": tests_path,
        "benchmarks": benchmarks_path,
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# --------------------------------------------------------------------------- #
# Running (I/O — the FD plumbing)
# --------------------------------------------------------------------------- #
# Map a run mode to which test-spec file it consumes.
_MODE_TESTFILE = {
    "test": "tests.txt",
    "benchmark": "benchmarks.txt",
    "leaderboard": "benchmarks.txt",
    "profile": "tests.txt",
}


def run_eval(
    workspace: Path,
    mode: str,
    problem: Problem,
    *,
    python_bin: str = "python",
    seed: int | None = None,
    timeout: int | None = None,
) -> EvalResult:
    """Run the staged eval harness in ``workspace`` and parse its output.

    The harness writes structured results to the fd named by ``POPCORN_FD`` (not
    stdout), so we hand it a temp-file fd and read it back afterward.
    """
    if mode not in _MODE_TESTFILE:
        raise LocalEvalError(f"unknown eval mode {mode!r}")
    testfile = workspace / _MODE_TESTFILE[mode]
    entry = workspace / problem.entry_point
    if not entry.exists():
        raise LocalEvalError(f"entry point {entry} not staged; call stage_workspace first")

    if timeout is None:
        timeout = problem.timeouts.get(f"{mode}_timeout") or problem.timeouts.get(
            "benchmark_timeout"
        )

    out_path = workspace / f".popcorn_fd_{mode}.txt"
    fd = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        env = dict(os.environ)
        env["POPCORN_FD"] = str(fd)
        if seed is not None:
            env["POPCORN_SEED"] = str(seed)
        # Ensure `from submission import ...` resolves even under spawn.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(workspace), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        try:
            proc = subprocess.run(
                [python_bin, problem.entry_point, mode, testfile.name],
                cwd=str(workspace),
                env=env,
                pass_fds=(fd,),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return EvalResult(
                mode=mode,
                exit_code=-1,
                passed=False,
                error=f"eval timed out after {timeout}s",
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
            )

        os.lseek(fd, 0, os.SEEK_SET)
        fd_text = os.read(fd, 10 * 1024 * 1024).decode("utf-8", errors="replace")
    finally:
        os.close(fd)

    return parse_eval_output(
        mode,
        fd_text,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
