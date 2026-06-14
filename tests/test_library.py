"""Offline tests for the knowledge library + its orchestrator wiring.

Exercises the lexical fallback index (chromadb not required), file persistence,
ingestion of update_library output, retrieval ranking, idempotency, and that a
run persists lessons through Orchestrator._persist_lessons.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_library
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from kernel_harness.library import Library, LibraryEntry, LexicalIndex
from kernel_harness.subagents import LibraryEntries, ProblemBrief


def test_lexical_index_ranks() -> None:
    idx = LexicalIndex()
    idx.add("a", "shared memory tiling for matmul improves locality")
    idx.add("b", "vectorized float4 loads reduce memory transactions")
    idx.add("c", "warp shuffle reduction for sum kernels")
    hits = idx.query("matmul tiling shared memory", k=2)
    assert hits and hits[0] == "a", hits
    assert idx.query("nonexistent terms zzz", k=3) == []
    print("ok  test_lexical_index_ranks")


def test_entry_id_is_content_hash() -> None:
    e1 = LibraryEntry("technique", "float4", "[technique] float4: vec loads")
    e2 = LibraryEntry("technique", "float4", "[technique] float4: vec loads")
    e3 = LibraryEntry("technique", "float4", "[technique] float4: different")
    assert e1.id == e2.id
    assert e1.id != e3.id
    print("ok  test_entry_id_is_content_hash")


def test_persist_and_query_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        lib = Library(td, prefer_chroma=False)
        assert lib.index_kind == "LexicalIndex"
        assert len(lib) == 0

        entries = LibraryEntries(
            techniques=[{"title": "float4 loads", "detail": "vectorize global loads", "applies_to": "memory-bound"}],
            failed_approaches=[{"approach": "naive elementwise", "why_failed": "uncoalesced accesses"}],
            winning_kernel={"approach": "triton tiled", "score": "2.3x", "notes": "BLOCK=128"},
        )
        ids = lib.persist_entries(entries, problem="vectoradd_v2", gpu="B200")
        assert len(ids) == 3
        assert len(lib) == 3

        # files written under kind dirs
        assert list((Path(td) / "technique").glob("*.json"))
        assert list((Path(td) / "winning_kernel").glob("*.json"))

        # reload from disk -> entries + index rebuilt
        lib2 = Library(td, prefer_chroma=False)
        assert len(lib2) == 3
        hits = lib2.query("memory-bound vectorize loads", k=3)
        assert any("float4" in h.text for h in hits), [h.text for h in hits]
        print("ok  test_persist_and_query_roundtrip")


def test_persist_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        lib = Library(td, prefer_chroma=False)
        entries = LibraryEntries(techniques=[{"title": "t", "detail": "d", "applies_to": "x"}])
        lib.persist_entries(entries, problem="p")
        lib.persist_entries(entries, problem="p")  # same content -> same id
        assert len(lib) == 1
        print("ok  test_persist_is_idempotent")


def test_candidates_for() -> None:
    with tempfile.TemporaryDirectory() as td:
        lib = Library(td, prefer_chroma=False)
        lib.persist_entries(
            LibraryEntries(techniques=[
                {"title": "float4 loads", "detail": "vectorize", "applies_to": "memory-bound add"},
                {"title": "warp reduce", "detail": "shuffle", "applies_to": "reductions"},
            ]),
            problem="vectoradd_v2",
        )
        brief = ProblemBrief(summary="float16 vector addition", dtype="float16",
                             optimization_targets=["memory bandwidth"])
        cands = lib.candidates_for(brief, k=2)
        assert cands and any("float4" in c for c in cands), cands
        # empty brief -> no query
        assert lib.candidates_for(ProblemBrief(), k=2) == []
        print("ok  test_candidates_for")


# --------------------------------------------------------------------------- #
def test_orchestrator_persists_lessons() -> None:
    """A run with a library calls update_library and persists entries."""
    from kernel_harness.agent import SubagentResult
    from kernel_harness.config import (
        AuthMode, BillingConfig, HardwareProfile, RunConfig, StoppingConditions,
    )
    from kernel_harness.orchestrator import EvalOutcome, Orchestrator, compute_speedup
    from kernel_harness.problem import Problem

    replies = {
        "problem_understander": '```json\n{"summary":"add","dtype":"float16","optimization_targets":["bandwidth"]}\n```',
        "kernel_writer": '```json\n{"approach":"triton","summary":"tiled","local_test_passed":true}\n```',
        "reflection": '```json\n{"action":"stop","reasoning":"good enough"}\n```',
        "profiler_interpreter": '```json\n{"bottleneck":"bandwidth","evidence":"x","recommendations":["y"]}\n```',
        "library_updater": '```json\n{"techniques":[{"title":"float4","detail":"vec","applies_to":"memory-bound"}],'
                           '"failed_approaches":[],"winning_kernel":{"approach":"triton","score":"2x","notes":"BLOCK=128"}}\n```',
    }

    class FakeRunner:
        async def run_subagent(self, name, prompt, **kwargs):
            return SubagentResult(name=name, model="m", text=replies[name], thinking="")

    class FakeFetcher:
        def fetch(self, name):  # noqa: ARG002
            return Problem(name="vectoradd_v2", problem_set="pmpp_v2",
                           directory="pmpp_v2/vectoradd_py", gpus=["B200"],
                           spec_files={"task.yml": "x"}, submission_filename="submission.py")

    with tempfile.TemporaryDirectory() as td:
        lib = Library(Path(td) / "lib", prefer_chroma=False)
        cfg = RunConfig(
            leaderboard="vectoradd_v2", gpu="B200",
            stop_on=StoppingConditions.parse("5iter"),
            hardware=HardwareProfile.BLACKWELL,
            billing=BillingConfig(mode=AuthMode.SUBSCRIPTION),
            candidate_approaches=["triton"],
        )
        orch = Orchestrator(cfg, runner=FakeRunner(), fetcher=FakeFetcher(),
                            runs_dir=Path(td) / "runs", library=lib)

        async def fake_baseline(problem):  # noqa: ARG001
            return 1000.0

        def fake_evaluate(record, problem, ref):  # noqa: ARG001
            return EvalOutcome(passed=True, geomean_ns=500.0, speedup=compute_speedup(ref, 500.0), source="local")

        orch._establish_baseline = fake_baseline  # type: ignore[assignment]
        orch._evaluate = fake_evaluate  # type: ignore[assignment]

        report = asyncio.run(orch.run())

    # reflection said 'stop' at iteration 2; library got the lesson persisted
    assert len(lib) >= 1, len(lib)
    assert any("float4" in e.text for e in lib._entries.values())
    print("ok  test_orchestrator_persists_lessons")


if __name__ == "__main__":
    test_lexical_index_ranks()
    test_entry_id_is_content_hash()
    test_persist_and_query_roundtrip()
    test_persist_is_idempotent()
    test_candidates_for()
    test_orchestrator_persists_lessons()
    print("\nall library checks passed")
