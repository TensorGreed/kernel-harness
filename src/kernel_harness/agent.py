"""Agent SDK wrapper — one ``query()`` per subagent, with billing fallback.

This is the spine the six subagents hang off. ``AgentRunner.run_subagent`` issues
a single Claude Agent SDK ``query()`` configured with:

* the **model** and **effort** routed for that subagent (see ``config.RunConfig``),
* **adaptive thinking**,
* whatever **tools** the caller scopes in (built-in Claude Code tools and/or our
  custom MCP tools for run-kernel / ncu / popcorn),
* an **auth/billing** env derived from ``BillingConfig``.

Billing fallback (the behavior the user asked for): runs start on the plan's
Agent SDK credit pool; when a query fails because that pool (or the plan's usage
limit) is exhausted, the runner switches to the configured pay-as-you-go API key
for the rest of the run and fires a one-time notice. If no key is available, it
raises ``CreditExhaustedError`` so the orchestrator can stop cleanly.

Everything here is async — ``query()`` is an async generator, and the
orchestrator + TUI are async too. See CONTEXT.md → "Tech Stack & Auth/Billing".
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    RateLimitEvent,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    query,
)

from .config import AuthMode, RunConfig


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class CreditExhaustedError(RuntimeError):
    """The current auth source can't pay and no fallback is available."""


class SubagentError(RuntimeError):
    """A subagent query failed for a non-billing reason (auth, invalid, server)."""

    def __init__(self, message: str, *, kind: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind


# Assistant-message error kinds that mean "this auth source can't pay — try the
# other one." A rejected rate limit on a subscription is the plan's usage cap;
# switching to pay-as-you-go is exactly the user's intent.
_BILLING_ERROR_KINDS = frozenset({"billing_error", "rate_limit"})
# HTTP statuses on a ResultMessage that indicate billing / rate exhaustion.
_BILLING_STATUSES = frozenset({402, 429})


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """A tool invocation observed in a subagent's transcript."""

    name: str
    input: dict[str, Any]


@dataclass
class SubagentResult:
    """The distilled outcome of one ``run_subagent`` call."""

    name: str
    model: str
    text: str                       # final text (ResultMessage.result or joined text)
    thinking: str                   # concatenated thinking blocks
    tool_calls: list[ToolCall] = field(default_factory=list)
    structured_output: Any = None   # populated when output_format was requested
    cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    num_turns: int = 0
    billed_to: AuthMode = AuthMode.SUBSCRIPTION
    backend: str = "cloud"          # "cloud" (Claude/Agent SDK) or "local" (Ollama/vLLM)


# --------------------------------------------------------------------------- #
# Exhaustion detection (pure helpers — unit-tested without a live query)
# --------------------------------------------------------------------------- #
def _rate_event_is_exhaustion(event: RateLimitEvent) -> bool:
    """True when a RateLimitEvent indicates the request was rejected/exhausted."""
    info = event.rate_limit_info
    if info.status == "rejected":
        return True
    # Subscription overage explicitly disabled / rejected -> credit pool is the cap.
    if info.overage_status == "rejected":
        return True
    if info.overage_disabled_reason:
        return True
    return False


def _result_is_exhaustion(result: ResultMessage | None) -> bool:
    """True when a terminal ResultMessage looks like a billing/rate failure."""
    if result is None or not result.is_error:
        return False
    if result.api_error_status in _BILLING_STATUSES:
        return True
    return False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class AgentRunner:
    """Issues Agent SDK queries for subagents, applying routing + billing policy."""

    def __init__(
        self,
        config: RunConfig,
        *,
        mcp_servers: dict[str, Any] | None = None,
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.billing = config.billing
        self._mcp_servers = mcp_servers or {}
        self._on_notice = on_notice or (lambda _msg: None)
        # Are we currently paying by API key? Pure-api-key mode starts that way.
        self._using_api_key = self.billing.mode is AuthMode.API_KEY

        # Warn early if the chosen mode can't actually pay.
        if self.billing.api_key_missing and self.billing.mode is AuthMode.API_KEY:
            self._on_notice(
                "api_key billing mode selected but no API key configured — "
                "queries will fail until one is provided."
            )
        # An ambient ANTHROPIC_API_KEY would silently force pay-per-token even in
        # subscription mode. Flag it so the user isn't surprised by charges.
        if self.billing.starts_on_subscription and os.environ.get("ANTHROPIC_API_KEY"):
            self._on_notice(
                "ANTHROPIC_API_KEY is set in the environment; the Claude CLI may "
                "use pay-per-token instead of your subscription credit. Unset it "
                "to run on subscription credit."
            )

    @property
    def current_auth(self) -> AuthMode:
        return AuthMode.API_KEY if self._using_api_key else AuthMode.SUBSCRIPTION

    # ----------------------------------------------------------------- #
    # Auth / options
    # ----------------------------------------------------------------- #
    def _auth_env(self, *, use_api_key: bool) -> dict[str, str]:
        """Env overrides handed to the SDK to select the billing source.

        For pay-as-you-go we inject the API key. For subscription we leave the
        key untouched and rely on the CLI's logged-in plan (see the ambient-key
        warning in ``__init__``).
        """
        if use_api_key:
            if not self.billing.api_key:
                raise SubagentError(
                    "pay-as-you-go requested but no API key configured", kind="config"
                )
            return {"ANTHROPIC_API_KEY": self.billing.api_key}
        return {}

    def build_options(
        self,
        name: str,
        *,
        system_prompt: str | None = None,
        allowed_tools: list[str] | None = None,
        effort: str | None = None,
        output_format: dict[str, Any] | None = None,
        max_turns: int | None = None,
        permission_mode: str = "bypassPermissions",
        mcp_servers: dict[str, Any] | None = None,
        use_api_key: bool | None = None,
    ) -> ClaudeAgentOptions:
        """Assemble ``ClaudeAgentOptions`` for a subagent (also unit-tested).

        ``mcp_servers`` overrides the runner-level default for this call — used to
        give parallel candidates their own isolated workspace tool servers.
        """
        if use_api_key is None:
            use_api_key = self._using_api_key

        opts = ClaudeAgentOptions(
            model=self.config.model_for(name),
            effort=effort or self.config.effort_for(name),  # type: ignore[arg-type]
            thinking={"type": "adaptive"},
            system_prompt=system_prompt,
            mcp_servers=mcp_servers if mcp_servers is not None else self._mcp_servers,
            permission_mode=permission_mode,  # type: ignore[arg-type]
            env=self._auth_env(use_api_key=use_api_key),
        )
        if allowed_tools is not None:
            opts.allowed_tools = allowed_tools
        if max_turns is not None:
            opts.max_turns = max_turns
        if output_format is not None:
            opts.output_format = output_format
        if self.billing.max_usd_per_query is not None:
            opts.max_budget_usd = self.billing.max_usd_per_query
        return opts

    # ----------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------- #
    async def run_subagent(
        self,
        name: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        allowed_tools: list[str] | None = None,
        effort: str | None = None,
        output_format: dict[str, Any] | None = None,
        max_turns: int | None = None,
        permission_mode: str = "bypassPermissions",
        mcp_servers: dict[str, Any] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_event: Callable[[Any], None] | None = None,
    ) -> SubagentResult:
        """Run a single subagent to completion, with one billing fallback.

        ``on_text`` receives streamed assistant text; ``on_event`` receives every
        raw SDK message (for the TUI). ``mcp_servers`` overrides the runner default
        for this call. Raises ``CreditExhaustedError`` if the loop runs out of ways
        to pay, or ``SubagentError`` on a non-billing failure.
        """
        kwargs = dict(
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            effort=effort,
            output_format=output_format,
            max_turns=max_turns,
            permission_mode=permission_mode,
            mcp_servers=mcp_servers,
            on_text=on_text,
            on_event=on_event,
        )
        try:
            return await self._attempt(name, prompt, use_api_key=self._using_api_key, **kwargs)
        except CreditExhaustedError:
            # Already on API key, or no fallback configured -> propagate.
            if self._using_api_key or not self.billing.can_fall_back_to_api_key:
                raise
            self._using_api_key = True
            self._on_notice(
                "Subscription Agent SDK credit exhausted — switching to pay-as-you-go "
                "API key for the remainder of this run."
            )
            return await self._attempt(name, prompt, use_api_key=True, **kwargs)

    async def _attempt(
        self,
        name: str,
        prompt: str,
        *,
        use_api_key: bool,
        system_prompt: str | None,
        allowed_tools: list[str] | None,
        effort: str | None,
        output_format: dict[str, Any] | None,
        max_turns: int | None,
        permission_mode: str,
        mcp_servers: dict[str, Any] | None,
        on_text: Callable[[str], None] | None,
        on_event: Callable[[Any], None] | None,
    ) -> SubagentResult:
        opts = self.build_options(
            name,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            effort=effort,
            output_format=output_format,
            max_turns=max_turns,
            permission_mode=permission_mode,
            mcp_servers=mcp_servers,
            use_api_key=use_api_key,
        )
        model = self.config.model_for(name)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        result: ResultMessage | None = None
        billing_hit = False

        async for msg in query(prompt=prompt, options=opts):
            if on_event is not None:
                on_event(msg)

            if isinstance(msg, AssistantMessage):
                if msg.error in _BILLING_ERROR_KINDS:
                    billing_hit = True
                elif msg.error is not None:
                    raise SubagentError(
                        f"subagent {name!r} failed: {msg.error}", kind=msg.error
                    )
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                        if on_text is not None:
                            on_text(block.text)
                    elif isinstance(block, ThinkingBlock):
                        thinking_parts.append(block.thinking)
                    elif isinstance(block, ToolUseBlock):
                        tool_calls.append(ToolCall(name=block.name, input=block.input))

            elif isinstance(msg, RateLimitEvent):
                if _rate_event_is_exhaustion(msg):
                    billing_hit = True

            elif isinstance(msg, ResultMessage):
                result = msg

        # Billing exhaustion -> let run_subagent decide whether to fall back.
        if billing_hit or _result_is_exhaustion(result):
            raise CreditExhaustedError(
                f"auth source {'api_key' if use_api_key else 'subscription'} "
                f"exhausted while running subagent {name!r}"
            )

        # Non-billing hard error on the terminal message.
        if result is not None and result.is_error:
            detail = "; ".join(result.errors or []) or result.subtype
            raise SubagentError(f"subagent {name!r} failed: {detail}")

        text = (result.result if result and result.result else "".join(text_parts)).strip()
        return SubagentResult(
            name=name,
            model=model,
            text=text,
            thinking="".join(thinking_parts).strip(),
            tool_calls=tool_calls,
            structured_output=result.structured_output if result else None,
            cost_usd=result.total_cost_usd if result else None,
            usage=result.usage if result else None,
            num_turns=result.num_turns if result else 0,
            billed_to=AuthMode.API_KEY if use_api_key else AuthMode.SUBSCRIPTION,
        )
