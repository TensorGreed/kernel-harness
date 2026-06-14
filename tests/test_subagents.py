"""Offline tests for the six subagents.

A fake runner returns canned JSON, so no model is called. We verify JSON
recovery, typed-result construction, prompt content, tool/system-prompt routing,
the empty-library short-circuit, and mcp_servers pass-through.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_subagents
"""

from __future__ import annotations

import asyncio

from kernel_harness.agent import AuthMode, SubagentResult
from kernel_harness import subagents as sa
from kernel_harness.problem import Problem


class FakeRunner:
    """Stands in for AgentRunner; records calls, returns scripted JSON text."""

    def __init__(self, replies: dict[str, str]) -> None:
        self._replies = replies
        self.calls: list[dict] = []

    async def run_subagent(self, name, prompt, **kwargs):
        self.calls.append({"name": name, "prompt": prompt, **kwargs})
        return SubagentResult(name=name, model="m", text=self._replies[name], thinking="")


def _brief() -> sa.ProblemBrief:
    return sa.ProblemBrief(summary="vector add", dtype="float16", kernel_signature="custom_kernel(a,b)->c")


def _problem() -> Problem:
    return Problem(
        name="vectoradd_v2", problem_set="pmpp_v2", directory="pmpp_v2/vectoradd_py",
        gpus=["B200"], spec_files={"task.yml": "description: add", "reference.py": "ref"},
    )


# --------------------------------------------------------------------------- #
def test_parse_json_block() -> None:
    assert sa.parse_json_block('```json\n{"a": 1}\n```')["a"] == 1
    assert sa.parse_json_block('prose\n```\n{"b": 2}\n```\nmore')["b"] == 2
    assert sa.parse_json_block('{"c": 3}')["c"] == 3
    assert sa.parse_json_block('here it is: {"d": 4} done')["d"] == 4
    try:
        sa.parse_json_block("no json here")
    except sa.SubagentParseError:
        pass
    else:
        raise AssertionError("expected SubagentParseError")
    print("ok  test_parse_json_block")


def test_understand_problem() -> None:
    reply = '```json\n{"summary":"add two fp16 tensors","dtype":"float16",' \
            '"kernel_signature":"custom_kernel(a,b)","constraints":["N x N"],' \
            '"optimization_targets":["bandwidth"]}\n```'
    runner = FakeRunner({"problem_understander": reply})
    brief = asyncio.run(sa.understand_problem(runner, _problem()))
    assert brief.dtype == "float16"
    assert brief.constraints == ["N x N"]
    assert brief.optimization_targets == ["bandwidth"]
    # routed correctly: right system prompt, no tools, problem context in prompt
    call = runner.calls[0]
    assert call["name"] == "problem_understander"
    assert call["allowed_tools"] == []
    assert "vectoradd_v2" in call["prompt"]
    print("ok  test_understand_problem")


def test_retrieve_knowledge_empty_shortcircuits() -> None:
    runner = FakeRunner({})  # would KeyError if called
    k = asyncio.run(sa.retrieve_knowledge(runner, _brief(), []))
    assert k.techniques == []
    assert "no prior" in k.rationale
    assert runner.calls == []  # never called the model
    print("ok  test_retrieve_knowledge_empty_shortcircuits")


def test_retrieve_knowledge_with_candidates() -> None:
    reply = '```json\n{"techniques":["use float4 vectorized loads"],' \
            '"rationale":"memory bound","pitfalls":["watch alignment"]}\n```'
    runner = FakeRunner({"library_retrieval": reply})
    k = asyncio.run(sa.retrieve_knowledge(runner, _brief(), ["float4 loads helped matmul"]))
    assert k.techniques == ["use float4 vectorized loads"]
    assert k.pitfalls == ["watch alignment"]
    assert "float4 loads helped matmul" in runner.calls[0]["prompt"]
    print("ok  test_retrieve_knowledge_with_candidates")


def test_write_kernel_passes_tools_and_mcp() -> None:
    reply = '```json\n{"approach":"triton","summary":"tiled add","local_test_passed":true}\n```'
    runner = FakeRunner({"kernel_writer": reply})
    servers = {"kernel_tools": {"type": "sdk"}}
    allowed = ["mcp__kernel_tools__save_submission", "mcp__kernel_tools__run_local"]
    cand = asyncio.run(
        sa.write_kernel(
            runner, _brief(), approach="triton",
            allowed_tools=allowed, mcp_servers=servers, gpu="B200",
            knowledge=sa.RetrievedKnowledge(techniques=["coalesce loads"]),
        )
    )
    assert cand.approach == "triton"
    assert cand.claimed_passing is True
    call = runner.calls[0]
    assert call["allowed_tools"] == allowed
    assert call["mcp_servers"] == servers
    assert "triton" in call["prompt"]
    assert "coalesce loads" in call["prompt"]
    assert "B200" in call["prompt"]
    print("ok  test_write_kernel_passes_tools_and_mcp")


def test_interpret_profile() -> None:
    reply = '```json\n{"bottleneck":"memory bandwidth","evidence":"DRAM 95%",' \
            '"recommendations":["vectorize loads","increase occupancy"]}\n```'
    runner = FakeRunner({"profiler_interpreter": reply})
    f = asyncio.run(
        sa.interpret_profile(runner, _brief(), allowed_tools=["mcp__kernel_tools__profile_kernel"], gpu="B200")
    )
    assert f.bottleneck == "memory bandwidth"
    assert len(f.recommendations) == 2
    print("ok  test_interpret_profile")


def test_reflect_decision() -> None:
    reply = '```json\n{"action":"submit","reasoning":"2.3x reference, plateaued",' \
            '"next_approach":"","focus":""}\n```'
    runner = FakeRunner({"reflection": reply})
    d = asyncio.run(sa.reflect(runner, _brief(), "iter1: 1.1x; iter2: 2.3x"))
    assert d.action == "submit"
    assert "plateaued" in d.reasoning
    # invalid action falls back to iterate
    runner2 = FakeRunner({"reflection": '```json\n{"action":"banana"}\n```'})
    d2 = asyncio.run(sa.reflect(runner2, _brief(), "h"))
    assert d2.action == "iterate"
    print("ok  test_reflect_decision")


def test_update_library() -> None:
    reply = '```json\n{"techniques":[{"title":"float4","detail":"vec loads","applies_to":"memory-bound"}],' \
            '"failed_approaches":[{"approach":"naive","why_failed":"uncoalesced"}],' \
            '"winning_kernel":{"approach":"triton","score":"2.3x","notes":"tiled"}}\n```'
    runner = FakeRunner({"library_updater": reply})
    e = asyncio.run(sa.update_library(runner, _brief(), "won at 2.3x with triton"))
    assert e.techniques[0]["title"] == "float4"
    assert e.failed_approaches[0]["approach"] == "naive"
    assert e.winning_kernel["score"] == "2.3x"
    print("ok  test_update_library")


if __name__ == "__main__":
    test_parse_json_block()
    test_understand_problem()
    test_retrieve_knowledge_empty_shortcircuits()
    test_retrieve_knowledge_with_candidates()
    test_write_kernel_passes_tools_and_mcp()
    test_interpret_profile()
    test_reflect_decision()
    test_update_library()
    print("\nall subagent checks passed")
