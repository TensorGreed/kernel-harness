"""Run configuration: hardware profiles, model routing, and stopping conditions.

The harness is parameterized at startup by a ``RunConfig``. Two pieces are
non-trivial and live here:

* ``StoppingConditions`` — parsed from the ``--stop-on`` flag (e.g.
  ``"2h|20iter|2x_reference"``). Conditions combine with OR semantics: the loop
  stops as soon as *any* one is met (whichever comes first).
* ``HardwareProfile`` — auto-detected via ``nvidia-smi`` at startup, overridable
  with ``--hardware``. Controls whether local testing and ncu/nsys profiling are
  available, and which spec sheet the kernel-writer and profiler subagents read.

See CONTEXT.md → "Hardware Profiles" and "Stopping Conditions".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# Hardware profiles
# --------------------------------------------------------------------------- #
class HardwareProfile(str, Enum):
    """Which capabilities the local machine offers to the loop."""

    BLACKWELL = "blackwell"          # local test + ncu w/ Blackwell spec sheet
    NVIDIA = "nvidia"                # local test + generic ncu
    LEADERBOARD_ONLY = "leaderboard-only"  # popcorn output only, no local profiling

    @property
    def can_run_locally(self) -> bool:
        """True when kernels can be executed on a local GPU before submitting."""
        return self is not HardwareProfile.LEADERBOARD_ONLY

    @property
    def can_profile(self) -> bool:
        """True when ncu/nsys profiling is available locally."""
        return self is not HardwareProfile.LEADERBOARD_ONLY


# Substrings (lowercased) that identify a Blackwell-class GPU from nvidia-smi.
_BLACKWELL_MARKERS = ("gb10", "b200", "b100", "rtx 50", "blackwell", "dgx spark")


def detect_hardware_profile() -> HardwareProfile:
    """Auto-detect the hardware profile from ``nvidia-smi``.

    Returns ``LEADERBOARD_ONLY`` when no NVIDIA GPU is visible (CPU box or a
    non-NVIDIA accelerator), ``BLACKWELL`` when the GPU name matches a known
    Blackwell marker, and ``NVIDIA`` otherwise. Callers can override the result
    via the ``--hardware`` flag.
    """
    if shutil.which("nvidia-smi") is None:
        return HardwareProfile.LEADERBOARD_ONLY

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return HardwareProfile.LEADERBOARD_ONLY

    if out.returncode != 0 or not out.stdout.strip():
        return HardwareProfile.LEADERBOARD_ONLY

    name = out.stdout.strip().splitlines()[0].lower()
    if any(marker in name for marker in _BLACKWELL_MARKERS):
        return HardwareProfile.BLACKWELL
    return HardwareProfile.NVIDIA


# --------------------------------------------------------------------------- #
# Stopping conditions
# --------------------------------------------------------------------------- #
class StoppingConditionError(ValueError):
    """Raised when a ``--stop-on`` spec cannot be parsed."""


@dataclass
class StoppingConditions:
    """Combinable termination criteria for the optimization loop.

    Any field left ``None``/``False`` is inactive. With multiple active
    conditions, the loop stops the moment the *first* one is satisfied.
    """

    time_budget_seconds: float | None = None
    iteration_budget: int | None = None
    target_speedup: float | None = None   # multiple of the reference impl's speed
    manual: bool = False

    # token form -> handler
    _TIME_RE = re.compile(r"^(\d+(?:\.\d+)?)(h|m|s)$")
    _ITER_RE = re.compile(r"^(\d+)iter$")
    _SPEEDUP_RE = re.compile(r"^(\d+(?:\.\d+)?)x(?:_reference|_ref)?$")

    @classmethod
    def parse(cls, spec: str) -> "StoppingConditions":
        """Parse a ``--stop-on`` spec like ``"2h|20iter|2x_reference"``.

        Supported tokens (case-insensitive, separated by ``|``):

        * ``<N>h`` / ``<N>m`` / ``<N>s`` — wall-clock time budget
        * ``<N>iter`` — iteration-count budget
        * ``<N>x`` / ``<N>x_reference`` — stop at N× the reference speed
        * ``manual`` — never auto-stop; the user decides via the TUI

        An empty spec means manual-only.
        """
        conditions = cls()
        spec = (spec or "").strip()
        if not spec:
            conditions.manual = True
            return conditions

        for raw in spec.split("|"):
            token = raw.strip().lower()
            if not token:
                continue
            if token == "manual":
                conditions.manual = True
                continue

            if (m := cls._TIME_RE.match(token)):
                value, unit = float(m.group(1)), m.group(2)
                seconds = value * {"h": 3600, "m": 60, "s": 1}[unit]
                conditions.time_budget_seconds = seconds
                continue

            if (m := cls._ITER_RE.match(token)):
                conditions.iteration_budget = int(m.group(1))
                continue

            if (m := cls._SPEEDUP_RE.match(token)):
                conditions.target_speedup = float(m.group(1))
                continue

            raise StoppingConditionError(
                f"unrecognized stop-on token: {raw!r}. "
                "Expected forms like '2h', '30m', '20iter', '2x_reference', or 'manual'."
            )

        if not conditions.is_active:
            # A spec that parsed to nothing actionable behaves as manual.
            conditions.manual = True
        return conditions

    @property
    def is_active(self) -> bool:
        """True if at least one auto-stop condition is set (manual excluded)."""
        return any(
            (
                self.time_budget_seconds is not None,
                self.iteration_budget is not None,
                self.target_speedup is not None,
            )
        )

    def met(
        self,
        *,
        elapsed_seconds: float,
        iterations: int,
        best_speedup: float | None,
    ) -> str | None:
        """Return a human-readable reason if any condition is satisfied, else None.

        ``best_speedup`` is the best speed-over-reference observed so far (or
        ``None`` if no correct kernel has been benchmarked yet).
        """
        if (
            self.time_budget_seconds is not None
            and elapsed_seconds >= self.time_budget_seconds
        ):
            return f"time budget reached ({self.time_budget_seconds:.0f}s)"
        if self.iteration_budget is not None and iterations >= self.iteration_budget:
            return f"iteration budget reached ({self.iteration_budget})"
        if (
            self.target_speedup is not None
            and best_speedup is not None
            and best_speedup >= self.target_speedup
        ):
            return (
                f"target speedup reached ({best_speedup:.2f}x "
                f">= {self.target_speedup:.2f}x reference)"
            )
        return None


# --------------------------------------------------------------------------- #
# Model routing
# --------------------------------------------------------------------------- #
# Default per-subagent model assignment. Opus for the high-stakes reasoning
# roles, Sonnet for the orchestrator + retrieval, Haiku for mechanical work.
# See CONTEXT.md → "Agent Architecture".
DEFAULT_MODEL_ROUTING: dict[str, str] = {
    "orchestrator": "claude-sonnet-4-6",
    "problem_understander": "claude-haiku-4-5",
    "workload_inspector": "claude-sonnet-4-6",
    "library_retrieval": "claude-sonnet-4-6",
    "kernel_writer": "claude-opus-4-8",
    "profiler_interpreter": "claude-opus-4-8",
    "reflection": "claude-opus-4-8",
    "research": "claude-opus-4-8",
    "library_updater": "claude-haiku-4-5",
}

# Per-subagent reasoning effort (Agent SDK ``effort`` knob: low|medium|high|xhigh|max).
# Mechanical roles run cheap; the high-stakes reasoning roles run deep.
DEFAULT_EFFORT_ROUTING: dict[str, str] = {
    "orchestrator": "medium",
    "problem_understander": "low",
    "workload_inspector": "medium",
    "library_retrieval": "medium",
    "kernel_writer": "high",
    "profiler_interpreter": "high",
    "reflection": "high",
    "research": "high",
    "library_updater": "low",
}

# Per-subagent compute backend: "cloud" (Claude via the Agent SDK) or "local"
# (an OpenAI-compatible Ollama/vLLM endpoint on the GB10). The mechanical,
# tool-free roles default to local to save cloud credit; the tool-using and
# highest-stakes roles stay on Claude. Fully overridable per subagent. Local
# only takes effect when a ``LocalConfig`` is attached to the RunConfig.
DEFAULT_BACKEND_ROUTING: dict[str, str] = {
    "orchestrator": "cloud",
    "problem_understander": "local",
    "workload_inspector": "cloud",
    "library_retrieval": "local",
    "kernel_writer": "cloud",       # needs the MCP tool loop + top quality
    "profiler_interpreter": "cloud",  # needs tools (ncu/run_local)
    "reflection": "cloud",
    "research": "cloud",
    "library_updater": "local",
}


@dataclass
class LocalConfig:
    """An OpenAI-compatible local-model endpoint (Ollama or vLLM).

    Backend-agnostic: both expose ``/v1/chat/completions``. Ollama ignores the
    API key; vLLM may require one. ``model_routing`` overrides ``default_model``
    per subagent.
    """

    base_url: str = "http://localhost:11434/v1"   # Ollama default
    api_key: str = "ollama"                        # placeholder; vLLM may need a real one
    default_model: str = "qwen3:30b"
    model_routing: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0

    def model_for(self, subagent: str) -> str:
        return self.model_routing.get(subagent, self.default_model)


# --------------------------------------------------------------------------- #
# Auth / billing
# --------------------------------------------------------------------------- #
# The harness runs on the Claude Agent SDK, which drives the `claude` CLI. Auth
# therefore flows through the CLI:
#   * subscription login (`claude` logged into a Pro/Max plan) -> draws from the
#     plan's monthly Agent SDK credit pool (Max 5x = $100/mo).
#   * an ANTHROPIC_API_KEY injected into the SDK's env -> pay-per-token.
# See CONTEXT.md -> "Auth & Billing".
class AuthMode(str, Enum):
    """How the harness pays for model calls."""

    SUBSCRIPTION = "subscription"        # credit pool only; halt when exhausted
    API_KEY = "api_key"                  # pure pay-as-you-go
    SUBSCRIPTION_THEN_API_KEY = "subscription_then_api_key"  # credit first, then fall back


@dataclass
class BillingConfig:
    """Auth source + graceful fallback when the credit pool is exhausted.

    In ``SUBSCRIPTION_THEN_API_KEY`` mode the loop starts on the plan's credit
    pool; when a call fails because credits are exhausted, the harness switches
    to ``api_key`` for the remainder of the run and surfaces a loud notice in the
    TUI — so the user is never silently blocked nor silently billed.
    """

    mode: AuthMode = AuthMode.SUBSCRIPTION_THEN_API_KEY
    api_key: str | None = None           # required for any mode that can bill per-token
    # Optional hard ceiling per subagent query, passed through to the SDK's
    # max_budget_usd. None = no per-call cap.
    max_usd_per_query: float | None = None

    def __post_init__(self) -> None:
        needs_key = self.mode in (AuthMode.API_KEY, AuthMode.SUBSCRIPTION_THEN_API_KEY)
        if needs_key and not self.api_key:
            # Not fatal at construction: subscription_then_api_key can start on
            # credits and only needs the key at fallback time. We record the gap
            # so the orchestrator can warn early.
            self.api_key_missing = True
        else:
            self.api_key_missing = False

    @property
    def starts_on_subscription(self) -> bool:
        return self.mode in (AuthMode.SUBSCRIPTION, AuthMode.SUBSCRIPTION_THEN_API_KEY)

    @property
    def can_fall_back_to_api_key(self) -> bool:
        return self.mode is AuthMode.SUBSCRIPTION_THEN_API_KEY and bool(self.api_key)


@dataclass
class RunConfig:
    """Top-level configuration for a single optimization run."""

    leaderboard: str                      # canonical popcorn name, e.g. "vectoradd_v2"
    gpu: str                              # leaderboard target GPU, e.g. "B200"
    stop_on: StoppingConditions
    hardware: HardwareProfile
    billing: BillingConfig = field(default_factory=BillingConfig)
    model_routing: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_ROUTING))
    effort_routing: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EFFORT_ROUTING))
    # Per-subagent backend ("cloud"/"local") and the local endpoint (None = all cloud).
    backend_routing: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BACKEND_ROUTING))
    local: LocalConfig | None = None
    # Distinct approaches to explore in parallel on iteration 1.
    candidate_approaches: list[str] = field(
        default_factory=lambda: ["triton", "cuda-inline", "pytorch"]
    )
    # Whether to fire a ranked leaderboard submission automatically when the loop
    # finishes. Off by default — a ranked submission is outward-facing, so the
    # user (or TUI) confirms it. The loop still reports the best kernel.
    auto_submit: bool = False

    def model_for(self, subagent: str) -> str:
        """Resolve the model id for a named subagent, defaulting to Opus."""
        return self.model_routing.get(subagent, "claude-opus-4-8")

    def effort_for(self, subagent: str) -> str:
        """Resolve the reasoning effort for a named subagent, defaulting to high."""
        return self.effort_routing.get(subagent, "high")

    def backend_for(self, subagent: str) -> str:
        """Resolve a subagent's backend. Always 'cloud' when no local endpoint set."""
        if self.local is None:
            return "cloud"
        return self.backend_routing.get(subagent, "cloud")

    def uses_local(self) -> bool:
        """True if any subagent is routed to a configured local endpoint."""
        return self.local is not None and any(
            v == "local" for v in self.backend_routing.values()
        )
