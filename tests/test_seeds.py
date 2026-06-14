"""Tests for library seeding + the workload_inspector subagent.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_seeds
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kernel_harness.agent import SubagentResult
from kernel_harness import subagents as sa
from kernel_harness.library import Library
from kernel_harness.problem import Problem
from kernel_harness.seeds import SEED_ENTRIES, seed_library


def test_seed_library() -> None:
    with tempfile.TemporaryDirectory() as td:
        lib = Library(td, prefer_chroma=False)
        n = seed_library(lib)
        assert n == len(SEED_ENTRIES) >= 15, n
        assert len(lib) == len(SEED_ENTRIES)

        # idempotent: re-seeding adds nothing (content-hash ids)
        seed_library(lib)
        assert len(lib) == len(SEED_ENTRIES)

        # every entry is attributed
        for e in lib._entries.values():
            assert e.metadata.get("source") == "github.com/Dogacel/auto-gpu-kernel"
            assert e.metadata.get("license") == "Apache-2.0"

        # reload from disk and retrieve a transferable lesson
        lib2 = Library(td, prefer_chroma=False)
        hits = lib2.query("blackwell triton num_warps tuning", k=3)
        assert any("num_warps" in h.text for h in hits), [h.text for h in hits]
        print(f"ok  test_seed_library ({n} entries)")


def test_workload_profile_block() -> None:
    wp = sa.WorkloadProfile(
        summary="grid is SM-starved",
        exploitable_structure=["only 8 CTAs vs 148 SMs"],
        shortcuts=["use split-K / flash-decoding"],
    )
    block = wp.as_block()
    assert "split-K" in block and "SM-starved" in block
    # empty profile -> empty block (no noise in the prompt)
    assert sa.WorkloadProfile().as_block() == ""
    print("ok  test_workload_profile_block")


def test_inspect_workload() -> None:
    reply = ('```json\n{"summary":"mostly padded, small grid",'
             '"regimes":["small T","large T"],'
             '"exploitable_structure":["median token uses 33 of 2048 entries"],'
             '"shortcuts":["early-exit on padding","split-K for small T"],"notes":"x"}\n```')

    class FakeRunner:
        def __init__(self): self.calls = []
        async def run_subagent(self, name, prompt, **kw):
            self.calls.append({"name": name, "prompt": prompt, **kw})
            return SubagentResult(name=name, model="m", text=reply, thinking="")

    runner = FakeRunner()
    problem = Problem(name="vectoradd_v2", problem_set="pmpp_v2", directory="d",
                      gpus=["B200"], spec_files={"reference.py": "def generate_input(): ..."},
                      tests=[{"size": 127}], benchmarks=[{"size": 1024}])
    brief = sa.ProblemBrief(summary="add", dtype="float16")

    wp = asyncio.run(sa.inspect_workload(runner, problem, brief))
    assert "split-K for small T" in wp.shortcuts
    assert len(wp.regimes) == 2
    call = runner.calls[0]
    assert call["name"] == "workload_inspector"
    assert call["allowed_tools"] == []
    assert "1024" in call["prompt"]  # benchmark sizes surfaced

    # the workload feeds the kernel-writer prompt
    prompt = sa.build_write_prompt(brief, approach="triton", workload=wp)
    assert "split-K for small T" in prompt
    assert "early-exit on padding" in prompt
    print("ok  test_inspect_workload")


if __name__ == "__main__":
    test_seed_library()
    test_workload_profile_block()
    test_inspect_workload()
    print("\nall seeds + workload-inspector checks passed")
