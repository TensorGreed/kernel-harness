"""Offline tests for the CLI arg→config mapping + the event printer.

The full `run` path needs network/GPU/model; here we test the pure pieces and
that the event printer tolerates every event shape. A separate live smoke test
(`kernel-harness list`) runs in the harness, not here.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_cli
"""

from __future__ import annotations

from kernel_harness.cli import build_parser, build_run_config, _make_event_printer
from kernel_harness.config import AuthMode, HardwareProfile


def test_parse_run_defaults() -> None:
    args = build_parser().parse_args(["run", "-l", "vectoradd_v2"])
    assert args.command == "run"
    assert args.leaderboard == "vectoradd_v2"
    assert args.stop_on == "10iter"
    assert args.gpu is None
    assert args.auth_mode == "subscription_then_api_key"
    assert args.approaches == "triton,cuda-inline,pytorch"
    assert args.auto_submit is False
    print("ok  test_parse_run_defaults")


def test_parse_run_overrides() -> None:
    args = build_parser().parse_args([
        "run", "--leaderboard", "matmul_v2", "--stop-on", "2h|3x_reference",
        "--gpu", "H100", "--hardware", "nvidia", "--auth-mode", "api_key",
        "--approaches", "triton, cutlass", "--auto-submit", "--max-usd-per-query", "0.5",
    ])
    assert args.leaderboard == "matmul_v2"
    assert args.hardware == "nvidia"
    assert args.auth_mode == "api_key"
    assert args.auto_submit is True
    assert args.max_usd_per_query == 0.5
    print("ok  test_parse_run_overrides")


def test_build_run_config() -> None:
    args = build_parser().parse_args([
        "run", "-l", "matmul_v2", "--stop-on", "2x_reference",
        "--approaches", "triton, cutlass , ", "--auto-submit",
    ])
    cfg = build_run_config(
        args, gpu="B200", hardware=HardwareProfile.BLACKWELL, api_key="sk-ant-x"
    )
    assert cfg.leaderboard == "matmul_v2"
    assert cfg.gpu == "B200"
    assert cfg.hardware is HardwareProfile.BLACKWELL
    assert cfg.billing.mode is AuthMode.SUBSCRIPTION_THEN_API_KEY
    assert cfg.billing.api_key == "sk-ant-x"
    assert cfg.stop_on.target_speedup == 2.0
    assert cfg.candidate_approaches == ["triton", "cutlass"]  # whitespace/empties stripped
    assert cfg.auto_submit is True
    # model routing carried through
    assert cfg.model_for("kernel_writer") == "claude-opus-4-8"
    print("ok  test_build_run_config")


def test_build_run_config_local_defaults() -> None:
    args = build_parser().parse_args(["run", "-l", "vectoradd_v2"])
    cfg = build_run_config(args, gpu="B200", hardware=HardwareProfile.BLACKWELL, api_key=None)
    # local enabled by default with the default Ollama endpoint + mechanical routing
    assert cfg.local is not None
    assert cfg.local.base_url == "http://localhost:11434/v1"
    assert cfg.local.default_model == "qwen3:30b"
    assert cfg.backend_for("problem_understander") == "local"
    assert cfg.backend_for("kernel_writer") == "cloud"
    print("ok  test_build_run_config_local_defaults")


def test_build_run_config_local_overrides() -> None:
    # --no-local disables local entirely
    args = build_parser().parse_args(["run", "-l", "x", "--no-local"])
    cfg = build_run_config(args, gpu="B200", hardware=HardwareProfile.BLACKWELL, api_key=None)
    assert cfg.local is None
    assert cfg.backend_for("problem_understander") == "cloud"

    # --local-subagents overrides routing; tool-using roles forced cloud even if listed
    args2 = build_parser().parse_args([
        "run", "-l", "x", "--local-subagents", "reflection,kernel_writer",
        "--local-base-url", "http://localhost:8000/v1", "--local-model", "llama3.3",
    ])
    cfg2 = build_run_config(args2, gpu="B200", hardware=HardwareProfile.BLACKWELL, api_key=None)
    assert cfg2.local.base_url == "http://localhost:8000/v1"
    assert cfg2.local.default_model == "llama3.3"
    assert cfg2.backend_for("reflection") == "local"
    assert cfg2.backend_for("kernel_writer") == "cloud"  # tool-using -> never local
    assert cfg2.backend_for("problem_understander") == "cloud"  # not in the list
    print("ok  test_build_run_config_local_overrides")


def test_event_printer_handles_all_events() -> None:
    on_event = _make_event_printer()
    samples = [
        ("run_start", {"problem": "vectoradd_v2", "gpu": "B200"}),
        ("brief", {"summary": "add two tensors"}),
        ("baseline", {"reference_ns": 1234.5}),
        ("baseline", {"reference_ns": None}),
        ("iteration_start", {"index": 1}),
        ("candidate_evaluated", {"id": "i1-c0-triton", "approach": "triton", "passed": True, "speedup": 1.8}),
        ("candidate_evaluated", {"id": "i1-c1-cuda", "approach": "cuda-inline", "passed": False, "speedup": None}),
        ("candidate_failed", {"approach": "cutlass", "error": "boom"}),
        ("decision", {"action": "iterate", "focus": "vectorize"}),
        ("notice", {"message": "switched to pay-as-you-go"}),
        ("submitting", {"cmd": "popcorn submit ..."}),
        ("submitted", {"ok": True}),
        ("run_end", {"reason": "target speedup reached", "best": "i1-c0-triton", "best_speedup": 2.1}),
    ]
    for name, payload in samples:
        on_event(name, payload)  # must not raise
    print("ok  test_event_printer_handles_all_events")


if __name__ == "__main__":
    test_parse_run_defaults()
    test_parse_run_overrides()
    test_build_run_config()
    test_build_run_config_local_defaults()
    test_build_run_config_local_overrides()
    test_event_printer_handles_all_events()
    print("\nall cli checks passed")
