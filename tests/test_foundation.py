"""Smoke tests for the config + problem-fetcher foundation.

Run with: .venv/bin/python -m tests.test_foundation
These are plain asserts (no pytest dep yet). The problem-fetcher test hits the
public GitHub API and is skipped offline.
"""

from __future__ import annotations

import sys

from kernel_harness.config import (
    AuthMode,
    BillingConfig,
    HardwareProfile,
    StoppingConditions,
    StoppingConditionError,
    detect_hardware_profile,
)


def test_stopping_conditions_parse() -> None:
    sc = StoppingConditions.parse("2h|20iter|2x_reference")
    assert sc.time_budget_seconds == 7200, sc.time_budget_seconds
    assert sc.iteration_budget == 20, sc.iteration_budget
    assert sc.target_speedup == 2.0, sc.target_speedup
    assert sc.is_active

    # Units
    assert StoppingConditions.parse("30m").time_budget_seconds == 1800
    assert StoppingConditions.parse("90s").time_budget_seconds == 90
    assert StoppingConditions.parse("1.5x").target_speedup == 1.5
    assert StoppingConditions.parse("5x_ref").target_speedup == 5.0

    # Empty / manual
    assert StoppingConditions.parse("").manual
    assert StoppingConditions.parse("manual").manual
    assert not StoppingConditions.parse("manual").is_active

    # Bad token
    try:
        StoppingConditions.parse("banana")
    except StoppingConditionError:
        pass
    else:
        raise AssertionError("expected StoppingConditionError for 'banana'")

    print("ok  test_stopping_conditions_parse")


def test_stopping_conditions_met() -> None:
    sc = StoppingConditions.parse("2h|20iter|2x_reference")

    assert sc.met(elapsed_seconds=10, iterations=1, best_speedup=None) is None
    assert sc.met(elapsed_seconds=7200, iterations=1, best_speedup=None)
    assert sc.met(elapsed_seconds=10, iterations=20, best_speedup=None)
    assert sc.met(elapsed_seconds=10, iterations=1, best_speedup=2.5)
    # below target -> not met
    assert sc.met(elapsed_seconds=10, iterations=1, best_speedup=1.5) is None

    print("ok  test_stopping_conditions_met")


def test_hardware_detection_runs() -> None:
    profile = detect_hardware_profile()
    assert isinstance(profile, HardwareProfile)
    print(f"ok  test_hardware_detection_runs -> {profile.value}")


def test_billing_config() -> None:
    # Default: subscription-first with API-key fallback. No key yet -> flagged.
    b = BillingConfig()
    assert b.mode is AuthMode.SUBSCRIPTION_THEN_API_KEY
    assert b.starts_on_subscription
    assert b.api_key_missing
    assert not b.can_fall_back_to_api_key  # no key -> can't actually fall back

    # With a key, fallback is live.
    b2 = BillingConfig(api_key="sk-ant-xxx")
    assert not b2.api_key_missing
    assert b2.can_fall_back_to_api_key

    # Pure subscription never falls back.
    b3 = BillingConfig(mode=AuthMode.SUBSCRIPTION)
    assert b3.starts_on_subscription
    assert not b3.can_fall_back_to_api_key
    assert not b3.api_key_missing  # subscription mode needs no key

    # Pure api-key without a key is flagged.
    b4 = BillingConfig(mode=AuthMode.API_KEY)
    assert b4.api_key_missing
    assert not b4.starts_on_subscription

    print("ok  test_billing_config")


def test_problem_fetcher_live() -> None:
    try:
        from kernel_harness.problem import ProblemFetcher, ProblemNotFoundError
        import httpx  # noqa: F401
    except ImportError as exc:
        print(f"skip test_problem_fetcher_live (missing dep: {exc})")
        return

    try:
        with ProblemFetcher() as fetcher:
            prob = fetcher.fetch("vectoradd_v2")
    except Exception as exc:  # network or API issue
        print(f"skip test_problem_fetcher_live (network: {exc})")
        return

    assert prob.name == "vectoradd_v2", prob.name
    assert prob.problem_set == "pmpp_v2", prob.problem_set
    assert prob.directory == "pmpp_v2/vectoradd_py", prob.directory
    assert "B200" in prob.gpus, prob.gpus
    assert "reference.py" in prob.spec_files, list(prob.spec_files)
    assert len(prob.spec_files["reference.py"]) > 0
    print(f"ok  test_problem_fetcher_live -> {prob.directory}, gpus={prob.gpus}")


if __name__ == "__main__":
    test_stopping_conditions_parse()
    test_stopping_conditions_met()
    test_hardware_detection_runs()
    test_billing_config()
    test_problem_fetcher_live()
    print("\nall foundation checks passed")
    sys.exit(0)
