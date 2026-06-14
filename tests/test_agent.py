"""Offline tests for the Agent SDK wrapper (kernel_harness.agent).

The live ``query()`` is faked, so these run with no network, no `claude` CLI,
and no spend. They cover the parts that carry real logic: exhaustion detection,
per-auth option building, and the subscription -> api-key billing fallback.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_agent
"""

from __future__ import annotations

import asyncio

from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    TextBlock,
)

import kernel_harness.agent as agent_mod
from kernel_harness.agent import (
    AgentRunner,
    CreditExhaustedError,
    _rate_event_is_exhaustion,
    _result_is_exhaustion,
)
from kernel_harness.config import AuthMode, BillingConfig, RunConfig
from kernel_harness.config import HardwareProfile, StoppingConditions


def _make_config(billing: BillingConfig) -> RunConfig:
    return RunConfig(
        leaderboard="vectoradd_v2",
        gpu="B200",
        stop_on=StoppingConditions.parse("5iter"),
        hardware=HardwareProfile.BLACKWELL,
        billing=billing,
    )


def _rate_event(status: str, **info) -> RateLimitEvent:
    return RateLimitEvent(
        rate_limit_info=RateLimitInfo(status=status, **info),
        uuid="u",
        session_id="s",
    )


def _result(*, is_error: bool, api_error_status: int | None = None, result: str | None = "done"):
    return ResultMessage(
        subtype="success" if not is_error else "error",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="s",
        api_error_status=api_error_status,
        result=result,
        total_cost_usd=0.01,
    )


# --------------------------------------------------------------------------- #
def test_exhaustion_helpers() -> None:
    assert _rate_event_is_exhaustion(_rate_event("rejected"))
    assert _rate_event_is_exhaustion(_rate_event("allowed", overage_status="rejected"))
    assert _rate_event_is_exhaustion(
        _rate_event("allowed", overage_disabled_reason="credit pool exhausted")
    )
    assert not _rate_event_is_exhaustion(_rate_event("allowed"))
    assert not _rate_event_is_exhaustion(_rate_event("allowed_warning"))

    assert _result_is_exhaustion(_result(is_error=True, api_error_status=429))
    assert _result_is_exhaustion(_result(is_error=True, api_error_status=402))
    assert not _result_is_exhaustion(_result(is_error=True, api_error_status=500))
    assert not _result_is_exhaustion(_result(is_error=False))
    assert not _result_is_exhaustion(None)
    print("ok  test_exhaustion_helpers")


def test_build_options_routing_and_auth() -> None:
    cfg = _make_config(BillingConfig(mode=AuthMode.SUBSCRIPTION))
    runner = AgentRunner(cfg)

    # Subscription: no API key injected; model + effort routed per subagent.
    opts = runner.build_options("kernel_writer", use_api_key=False)
    assert opts.model == "claude-opus-4-8", opts.model
    assert opts.effort == "high", opts.effort
    assert opts.thinking == {"type": "adaptive"}, opts.thinking
    assert "ANTHROPIC_API_KEY" not in opts.env, opts.env

    opts_h = runner.build_options("problem_understander", use_api_key=False)
    assert opts_h.model == "claude-haiku-4-5"
    assert opts_h.effort == "low"

    # Pay-as-you-go: key injected into env.
    cfg2 = _make_config(BillingConfig(mode=AuthMode.API_KEY, api_key="sk-ant-test"))
    runner2 = AgentRunner(cfg2)
    opts2 = runner2.build_options("kernel_writer", use_api_key=True)
    assert opts2.env.get("ANTHROPIC_API_KEY") == "sk-ant-test"
    print("ok  test_build_options_routing_and_auth")


# --- fake query infrastructure --------------------------------------------- #
def _fake_query_factory(script):
    """Build a fake ``query`` that yields messages from ``script(use_api_key)``.

    ``script`` maps whether the call used an API key (inferred from options.env)
    to a list of SDK messages to yield.
    """

    async def fake_query(*, prompt, options, transport=None):  # noqa: ARG001
        use_api_key = bool(options.env.get("ANTHROPIC_API_KEY"))
        for msg in script(use_api_key):
            yield msg

    return fake_query


def test_billing_fallback(monkeypatch) -> None:
    """Subscription call hits a billing error -> runner falls back to the key."""

    def script(use_api_key: bool):
        if not use_api_key:
            # Subscription attempt: signal credit exhaustion.
            return [
                AssistantMessage(content=[TextBlock(text="")], model="m", error="billing_error"),
                _result(is_error=True, api_error_status=429, result=None),
            ]
        # API-key attempt succeeds.
        return [
            AssistantMessage(content=[TextBlock(text="hello from paid path")], model="m"),
            _result(is_error=False, result="hello from paid path"),
        ]

    monkeypatch.setattr(agent_mod, "query", _fake_query_factory(script))

    notices: list[str] = []
    cfg = _make_config(
        BillingConfig(mode=AuthMode.SUBSCRIPTION_THEN_API_KEY, api_key="sk-ant-test")
    )
    runner = AgentRunner(cfg, on_notice=notices.append)
    assert runner.current_auth is AuthMode.SUBSCRIPTION

    res = asyncio.run(runner.run_subagent("kernel_writer", "write a kernel"))

    assert res.text == "hello from paid path", res.text
    assert res.billed_to is AuthMode.API_KEY
    assert runner.current_auth is AuthMode.API_KEY  # sticky for rest of run
    assert any("switching to pay-as-you-go" in n for n in notices), notices
    print("ok  test_billing_fallback")


def test_no_fallback_raises(monkeypatch) -> None:
    """Subscription-only mode with no key -> exhaustion propagates."""

    def script(use_api_key: bool):  # noqa: ARG001
        return [
            AssistantMessage(content=[TextBlock(text="")], model="m", error="billing_error"),
            _result(is_error=True, api_error_status=429, result=None),
        ]

    monkeypatch.setattr(agent_mod, "query", _fake_query_factory(script))

    cfg = _make_config(BillingConfig(mode=AuthMode.SUBSCRIPTION))
    runner = AgentRunner(cfg)
    try:
        asyncio.run(runner.run_subagent("kernel_writer", "write a kernel"))
    except CreditExhaustedError:
        print("ok  test_no_fallback_raises")
    else:
        raise AssertionError("expected CreditExhaustedError with no fallback")


def test_happy_path(monkeypatch) -> None:
    """A clean subscription run returns text, cost, and tool calls."""

    def script(use_api_key: bool):  # noqa: ARG001
        return [
            AssistantMessage(content=[TextBlock(text="the answer")], model="m"),
            _result(is_error=False, result="the answer"),
        ]

    monkeypatch.setattr(agent_mod, "query", _fake_query_factory(script))

    cfg = _make_config(BillingConfig(mode=AuthMode.SUBSCRIPTION))
    runner = AgentRunner(cfg)
    res = asyncio.run(runner.run_subagent("reflection", "what next?"))
    assert res.text == "the answer"
    assert res.billed_to is AuthMode.SUBSCRIPTION
    assert res.cost_usd == 0.01
    print("ok  test_happy_path")


# --------------------------------------------------------------------------- #
# Minimal monkeypatch shim so we don't need pytest.
# --------------------------------------------------------------------------- #
class _MonkeyPatch:
    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target, name, value) -> None:
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo.clear()


if __name__ == "__main__":
    test_exhaustion_helpers()
    test_build_options_routing_and_auth()
    for fn in (test_billing_fallback, test_no_fallback_raises, test_happy_path):
        mp = _MonkeyPatch()
        try:
            fn(mp)
        finally:
            mp.undo()
    print("\nall agent checks passed")
