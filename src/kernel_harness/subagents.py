"""The six specialized subagents.

Each subagent is a thin async function over :class:`~kernel_harness.agent.AgentRunner`:
it builds a focused prompt, runs one ``query()`` with the model/effort routed for
its role, and parses a typed result out of the model's JSON output. The
orchestrator owns the loop; these are its moving parts.

| function            | role                  | model  | tools                         |
|---------------------|-----------------------|--------|-------------------------------|
| understand_problem  | problem_understander  | Haiku  | none (reads spec text)        |
| retrieve_knowledge  | library_retrieval     | Sonnet | none (reasons over candidates)|
| write_kernel        | kernel_writer         | Opus   | kernel_tools + Write/Read/Edit/Bash |
| interpret_profile   | profiler_interpreter  | Opus   | profile_kernel / run_local    |
| reflect             | reflection            | Opus   | none (reasons over history)   |
| update_library      | library_updater       | Haiku  | none (emits entries)          |

Results are parsed from a fenced ```json block (model-agnostic, offline-testable)
rather than relying on SDK structured-output wiring. See CONTEXT.md → "Agent
Architecture".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .agent import AgentRunner, SubagentResult
from .problem import Problem


class SubagentParseError(RuntimeError):
    """Raised when a subagent's JSON payload can't be recovered."""


# --------------------------------------------------------------------------- #
# JSON extraction (pure, offline-tested)
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def parse_json_block(text: str) -> dict:
    """Recover a JSON object from a model reply.

    Tries, in order: a fenced ```json block, the whole string, and the widest
    ``{...}`` substring. Raises ``SubagentParseError`` if none parse.
    """
    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text)

    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i, j = chunk.find("{"), chunk.rfind("}")
        if 0 <= i < j:
            try:
                obj = json.loads(chunk[i : j + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    raise SubagentParseError("no JSON object found in subagent reply")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class ProblemBrief:
    summary: str = ""
    dtype: str = ""
    input_spec: str = ""
    output_spec: str = ""
    tolerance: str = ""
    kernel_signature: str = ""        # what custom_kernel must accept/return
    constraints: list[str] = field(default_factory=list)
    optimization_targets: list[str] = field(default_factory=list)
    raw: SubagentResult | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemBrief":
        return cls(
            summary=str(d.get("summary", "")),
            dtype=str(d.get("dtype", "")),
            input_spec=str(d.get("input_spec", "")),
            output_spec=str(d.get("output_spec", "")),
            tolerance=str(d.get("tolerance", "")),
            kernel_signature=str(d.get("kernel_signature", "")),
            constraints=[str(x) for x in _as_list(d.get("constraints"))],
            optimization_targets=[str(x) for x in _as_list(d.get("optimization_targets"))],
        )


@dataclass
class WorkloadProfile:
    """Structure in the problem's *input distribution* worth exploiting.

    Analyzing inputs (shapes, sparsity, padding, value ranges) for regime-specific
    shortcuts is often the single biggest structural lever — e.g. an SM-starved
    grid that wants split-K, or a mostly-padded input that wants an early exit.
    """

    summary: str = ""
    regimes: list[str] = field(default_factory=list)            # distinct input regimes
    exploitable_structure: list[str] = field(default_factory=list)
    shortcuts: list[str] = field(default_factory=list)          # concrete, actionable
    notes: str = ""
    raw: SubagentResult | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "WorkloadProfile":
        return cls(
            summary=str(d.get("summary", "")),
            regimes=[str(x) for x in _as_list(d.get("regimes"))],
            exploitable_structure=[str(x) for x in _as_list(d.get("exploitable_structure"))],
            shortcuts=[str(x) for x in _as_list(d.get("shortcuts"))],
            notes=str(d.get("notes", "")),
        )

    def as_block(self) -> str:
        if not (self.exploitable_structure or self.shortcuts):
            return ""
        parts = ["\nWorkload structure to exploit:"]
        if self.summary:
            parts.append(f"- {self.summary}")
        for s in self.exploitable_structure:
            parts.append(f"- structure: {s}")
        for s in self.shortcuts:
            parts.append(f"- shortcut: {s}")
        return "\n".join(parts)


@dataclass
class RetrievedKnowledge:
    techniques: list[str] = field(default_factory=list)
    rationale: str = ""
    pitfalls: list[str] = field(default_factory=list)
    raw: SubagentResult | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievedKnowledge":
        return cls(
            techniques=[str(x) for x in _as_list(d.get("techniques"))],
            rationale=str(d.get("rationale", "")),
            pitfalls=[str(x) for x in _as_list(d.get("pitfalls"))],
        )

    @classmethod
    def empty(cls) -> "RetrievedKnowledge":
        return cls(rationale="no prior library entries available")


@dataclass
class KernelCandidate:
    approach: str = ""               # e.g. "triton", "cuda-inline", "cutlass"
    summary: str = ""                # what the writer did / key decisions
    claimed_passing: bool | None = None  # the writer's self-reported local test
    raw: SubagentResult | None = None

    @classmethod
    def from_dict(cls, d: dict, *, approach: str) -> "KernelCandidate":
        return cls(
            approach=str(d.get("approach", approach)),
            summary=str(d.get("summary", "")),
            claimed_passing=d.get("local_test_passed"),
        )


@dataclass
class ProfilerFindings:
    bottleneck: str = ""
    evidence: str = ""
    recommendations: list[str] = field(default_factory=list)
    raw: SubagentResult | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ProfilerFindings":
        return cls(
            bottleneck=str(d.get("bottleneck", "")),
            evidence=str(d.get("evidence", "")),
            recommendations=[str(x) for x in _as_list(d.get("recommendations"))],
        )


@dataclass
class StrategyDecision:
    action: str = "iterate"          # "iterate" | "submit" | "stop"
    reasoning: str = ""
    next_approach: str = ""
    focus: str = ""                  # what the next rewrite should target
    raw: SubagentResult | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyDecision":
        action = str(d.get("action", "iterate")).lower()
        if action not in ("iterate", "submit", "stop"):
            action = "iterate"
        return cls(
            action=action,
            reasoning=str(d.get("reasoning", "")),
            next_approach=str(d.get("next_approach", "")),
            focus=str(d.get("focus", "")),
        )


@dataclass
class LibraryEntries:
    techniques: list[dict] = field(default_factory=list)
    failed_approaches: list[dict] = field(default_factory=list)
    winning_kernel: dict | None = None
    raw: SubagentResult | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LibraryEntries":
        return cls(
            techniques=[x for x in _as_list(d.get("techniques")) if isinstance(x, dict)],
            failed_approaches=[x for x in _as_list(d.get("failed_approaches")) if isinstance(x, dict)],
            winning_kernel=d.get("winning_kernel") if isinstance(d.get("winning_kernel"), dict) else None,
        )


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #
SYSTEM_PROMPTS: dict[str, str] = {
    "problem_understander": (
        "You analyze a GPU MODE kernel problem and extract a precise, structured "
        "brief for downstream kernel-writing agents. Read the reference "
        "implementation, task schema, and description carefully. Be exact about "
        "dtype, tensor shapes, the custom_kernel signature, and the correctness "
        "tolerance. Do not write a kernel. Respond with a single JSON object in a "
        "```json code block with keys: summary, dtype, input_spec, output_spec, "
        "tolerance, kernel_signature, constraints (list), optimization_targets (list)."
    ),
    "workload_inspector": (
        "You analyze the INPUT DISTRIBUTION of a GPU MODE problem to find "
        "regime-specific shortcuts a kernel can exploit. Study the reference "
        "implementation, the input generator (generate_input), and the test/benchmark "
        "sizes. Look for: small grids that leave SMs idle (want split-K/flash-decoding), "
        "heavy padding or sparsity (want early-exit), size-1 axes or trivializing shapes, "
        "bimodal size regimes that want per-regime dispatch, contiguous/aligned structure, "
        "and value ranges affecting numerics. Be concrete and quantitative where the "
        "sizes allow. Respond with a single JSON object in a ```json code block with keys: "
        "summary, regimes (list), exploitable_structure (list), shortcuts (list of concrete "
        "actionable strings), notes."
    ),
    "library_retrieval": (
        "You are a GPU optimization librarian. Given a problem brief and a set of "
        "candidate notes from prior hackathons (techniques, winning kernels, failed "
        "approaches), select what is genuinely relevant to THIS problem and explain "
        "how to apply it. Ignore irrelevant entries. If no candidates are useful, "
        "say so. Respond with a single JSON object in a ```json code block with "
        "keys: techniques (list of concrete, actionable strings), rationale, "
        "pitfalls (list)."
    ),
    "kernel_writer": (
        "You are an elite GPU kernel engineer competing on the GPU MODE leaderboard. "
        "Write a single-file Python submission exposing a `custom_kernel` function "
        "matching the required signature. You may use inline CUDA C++ "
        "(torch.utils.cpp_extension.load_inline), Triton, PyTorch, CUTLASS, "
        "ThunderKittens, or whatever the task allows. Correctness first, then speed. "
        "Use the save_submission tool to write your code, then run_local('test') to "
        "verify correctness and run_local('benchmark') to measure timing. Iterate "
        "until it passes. When given profiler findings or a prior attempt, address "
        "the specific bottleneck rather than rewriting blindly. End with a single "
        "JSON object in a ```json code block with keys: approach, summary, "
        "local_test_passed (boolean)."
    ),
    "profiler_interpreter": (
        "You are a GPU performance analyst. Profile the current submission with the "
        "profile_kernel tool (Nsight Compute) and, if helpful, run_local('benchmark'). "
        "Identify the single most important bottleneck (memory bandwidth, occupancy, "
        "compute, latency, etc.) with concrete evidence from the report, and give "
        "specific, actionable rewrite recommendations the kernel writer can act on. "
        "Reason about the target GPU's real limits, not generic advice. Respond with "
        "a single JSON object in a ```json code block with keys: bottleneck, evidence, "
        "recommendations (list)."
    ),
    "reflection": (
        "You are the strategist for an autonomous kernel-optimization loop. Given the "
        "full history of attempts (approaches tried, correctness, local and "
        "leaderboard timings, profiler findings), decide what to do next to win. "
        "Choose 'iterate' (keep optimizing — say what to focus on next), 'submit' "
        "(the current best is ready for a ranked leaderboard submission), or 'stop' "
        "(diminishing returns or blocked). Be decisive and specific. Respond with a "
        "single JSON object in a ```json code block with keys: action "
        "('iterate'|'submit'|'stop'), reasoning, next_approach, focus."
    ),
    "library_updater": (
        "You curate a cross-hackathon knowledge library. Given the outcome of a run, "
        "distill durable, reusable lessons: techniques that worked (with the context "
        "they apply to), approaches that failed (and why), and the winning kernel if "
        "any. Be concrete and generalizable; skip anything trivial or problem-"
        "specific that won't transfer. Respond with a single JSON object in a ```json "
        "code block with keys: techniques (list of {title, detail, applies_to}), "
        "failed_approaches (list of {approach, why_failed}), winning_kernel "
        "({approach, score, notes} or null)."
    ),
}


# --------------------------------------------------------------------------- #
# Prompt builders (pure)
# --------------------------------------------------------------------------- #
def _brief_block(brief: ProblemBrief) -> str:
    return (
        f"Problem brief:\n"
        f"- summary: {brief.summary}\n"
        f"- dtype: {brief.dtype}\n"
        f"- input: {brief.input_spec}\n"
        f"- output: {brief.output_spec}\n"
        f"- tolerance: {brief.tolerance}\n"
        f"- kernel_signature: {brief.kernel_signature}\n"
        f"- constraints: {', '.join(brief.constraints) or 'none'}\n"
        f"- optimization_targets: {', '.join(brief.optimization_targets) or 'none'}"
    )


def build_understand_prompt(problem: Problem) -> str:
    return (
        "Analyze this GPU MODE problem and produce the structured brief.\n\n"
        + problem.as_prompt_context()
    )


def build_inspect_prompt(problem: Problem, brief: ProblemBrief) -> str:
    sizes = ""
    if problem.tests or problem.benchmarks:
        sizes = (
            f"\nTest sizes: {problem.tests}\nBenchmark sizes: {problem.benchmarks}"
        )
    return (
        f"{_brief_block(brief)}\n\nAnalyze the input distribution for exploitable "
        f"structure.{sizes}\n\n{problem.as_prompt_context()}"
    )


def build_retrieve_prompt(brief: ProblemBrief, candidate_entries: list[str]) -> str:
    if candidate_entries:
        entries = "\n\n".join(f"[{i}] {e}" for i, e in enumerate(candidate_entries))
    else:
        entries = "(no prior library entries)"
    return f"{_brief_block(brief)}\n\nCandidate library notes:\n{entries}"


def build_write_prompt(
    brief: ProblemBrief,
    *,
    approach: str,
    knowledge: RetrievedKnowledge | None = None,
    gpu: str = "",
    hardware_notes: str = "",
    prior_summary: str = "",
    profiler: ProfilerFindings | None = None,
    workload: "WorkloadProfile | None" = None,
) -> str:
    parts = [_brief_block(brief)]
    parts.append(f"\nTarget GPU: {gpu or 'unspecified'}")
    if hardware_notes:
        parts.append(f"Hardware notes: {hardware_notes}")
    if workload is not None:
        block = workload.as_block()
        if block:
            parts.append(block)
    if approach:
        parts.append(f"\nUse this approach for this candidate: {approach}")
    if knowledge and knowledge.techniques:
        parts.append("\nRelevant techniques from prior work:\n- " + "\n- ".join(knowledge.techniques))
        if knowledge.pitfalls:
            parts.append("Known pitfalls:\n- " + "\n- ".join(knowledge.pitfalls))
    if prior_summary:
        parts.append(f"\nPrior attempt:\n{prior_summary}")
    if profiler:
        parts.append(
            f"\nProfiler findings to address:\n- bottleneck: {profiler.bottleneck}\n"
            f"- evidence: {profiler.evidence}\n- recommendations:\n  - "
            + "\n  - ".join(profiler.recommendations)
        )
    parts.append(
        "\nWrite the kernel now. Save it with save_submission, verify with "
        "run_local('test'), and benchmark with run_local('benchmark')."
    )
    return "\n".join(parts)


def build_profile_prompt(brief: ProblemBrief, *, gpu: str = "", hardware_notes: str = "") -> str:
    head = _brief_block(brief)
    hw = f"\nTarget GPU: {gpu or 'unspecified'}"
    if hardware_notes:
        hw += f"\nHardware notes: {hardware_notes}"
    return (
        f"{head}{hw}\n\nThe current submission passes correctness. Profile it and "
        "identify the most important bottleneck with concrete evidence and "
        "actionable rewrite recommendations."
    )


def build_reflect_prompt(brief: ProblemBrief, history: str) -> str:
    return (
        f"{_brief_block(brief)}\n\nIteration history so far:\n{history}\n\n"
        "Decide the next action to maximize leaderboard performance."
    )


def build_update_library_prompt(brief: ProblemBrief, outcome: str) -> str:
    return (
        f"{_brief_block(brief)}\n\nRun outcome:\n{outcome}\n\n"
        "Distill durable library entries from this run."
    )


# --------------------------------------------------------------------------- #
# Subagent runners
# --------------------------------------------------------------------------- #
async def understand_problem(runner: AgentRunner, problem: Problem) -> ProblemBrief:
    res = await runner.run_subagent(
        "problem_understander",
        build_understand_prompt(problem),
        system_prompt=SYSTEM_PROMPTS["problem_understander"],
        allowed_tools=[],
    )
    brief = ProblemBrief.from_dict(parse_json_block(res.text))
    brief.raw = res
    return brief


async def inspect_workload(
    runner: AgentRunner, problem: Problem, brief: ProblemBrief
) -> WorkloadProfile:
    res = await runner.run_subagent(
        "workload_inspector",
        build_inspect_prompt(problem, brief),
        system_prompt=SYSTEM_PROMPTS["workload_inspector"],
        allowed_tools=[],
    )
    profile = WorkloadProfile.from_dict(parse_json_block(res.text))
    profile.raw = res
    return profile


async def retrieve_knowledge(
    runner: AgentRunner,
    brief: ProblemBrief,
    candidate_entries: list[str],
) -> RetrievedKnowledge:
    if not candidate_entries:
        return RetrievedKnowledge.empty()
    res = await runner.run_subagent(
        "library_retrieval",
        build_retrieve_prompt(brief, candidate_entries),
        system_prompt=SYSTEM_PROMPTS["library_retrieval"],
        allowed_tools=[],
    )
    knowledge = RetrievedKnowledge.from_dict(parse_json_block(res.text))
    knowledge.raw = res
    return knowledge


async def write_kernel(
    runner: AgentRunner,
    brief: ProblemBrief,
    *,
    approach: str,
    allowed_tools: list[str],
    mcp_servers: dict | None = None,
    knowledge: RetrievedKnowledge | None = None,
    gpu: str = "",
    hardware_notes: str = "",
    prior_summary: str = "",
    profiler: ProfilerFindings | None = None,
    workload: WorkloadProfile | None = None,
    on_text=None,
) -> KernelCandidate:
    res = await runner.run_subagent(
        "kernel_writer",
        build_write_prompt(
            brief,
            approach=approach,
            knowledge=knowledge,
            gpu=gpu,
            hardware_notes=hardware_notes,
            prior_summary=prior_summary,
            profiler=profiler,
            workload=workload,
        ),
        system_prompt=SYSTEM_PROMPTS["kernel_writer"],
        allowed_tools=allowed_tools,
        mcp_servers=mcp_servers,
        on_text=on_text,
    )
    cand = KernelCandidate.from_dict(parse_json_block(res.text), approach=approach)
    cand.raw = res
    return cand


async def interpret_profile(
    runner: AgentRunner,
    brief: ProblemBrief,
    *,
    allowed_tools: list[str],
    mcp_servers: dict | None = None,
    gpu: str = "",
    hardware_notes: str = "",
    on_text=None,
) -> ProfilerFindings:
    res = await runner.run_subagent(
        "profiler_interpreter",
        build_profile_prompt(brief, gpu=gpu, hardware_notes=hardware_notes),
        system_prompt=SYSTEM_PROMPTS["profiler_interpreter"],
        allowed_tools=allowed_tools,
        mcp_servers=mcp_servers,
        on_text=on_text,
    )
    findings = ProfilerFindings.from_dict(parse_json_block(res.text))
    findings.raw = res
    return findings


async def reflect(runner: AgentRunner, brief: ProblemBrief, history: str) -> StrategyDecision:
    res = await runner.run_subagent(
        "reflection",
        build_reflect_prompt(brief, history),
        system_prompt=SYSTEM_PROMPTS["reflection"],
        allowed_tools=[],
    )
    decision = StrategyDecision.from_dict(parse_json_block(res.text))
    decision.raw = res
    return decision


async def update_library(runner: AgentRunner, brief: ProblemBrief, outcome: str) -> LibraryEntries:
    res = await runner.run_subagent(
        "library_updater",
        build_update_library_prompt(brief, outcome),
        system_prompt=SYSTEM_PROMPTS["library_updater"],
        allowed_tools=[],
    )
    entries = LibraryEntries.from_dict(parse_json_block(res.text))
    entries.raw = res
    return entries
