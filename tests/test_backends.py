"""Offline tests for the local-model backend + per-subagent router.

A mock httpx transport stands in for Ollama/vLLM, so no server is needed. We
verify the OpenAI-compatible call shape, response parsing, per-subagent
dispatch, and the cloud fallback when the local endpoint errors.

Run: PYTHONPATH=src .venv/bin/python -m tests.test_backends
"""

from __future__ import annotations

import asyncio

import httpx

from kernel_harness.agent import SubagentResult
from kernel_harness.backends import BackendRouter, LocalBackendError, LocalRunner
from kernel_harness.config import (
    DEFAULT_BACKEND_ROUTING,
    HardwareProfile,
    LocalConfig,
    RunConfig,
    StoppingConditions,
)


def _config(*, local: LocalConfig | None, routing: dict | None = None) -> RunConfig:
    return RunConfig(
        leaderboard="vectoradd_v2", gpu="B200",
        stop_on=StoppingConditions.parse("5iter"),
        hardware=HardwareProfile.BLACKWELL,
        local=local,
        backend_routing=routing or dict(DEFAULT_BACKEND_ROUTING),
    )


# --------------------------------------------------------------------------- #
def test_config_backend_for() -> None:
    # No local endpoint -> everything cloud regardless of routing.
    cfg_cloud = _config(local=None)
    assert cfg_cloud.backend_for("problem_understander") == "cloud"
    assert not cfg_cloud.uses_local()

    # With local endpoint, defaults route the mechanical roles local.
    cfg = _config(local=LocalConfig())
    assert cfg.backend_for("problem_understander") == "local"
    assert cfg.backend_for("library_updater") == "local"
    assert cfg.backend_for("kernel_writer") == "cloud"
    assert cfg.backend_for("profiler_interpreter") == "cloud"
    assert cfg.uses_local()
    print("ok  test_config_backend_for")


def test_local_runner_call_shape() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        import json

        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}],
                  "usage": {"total_tokens": 42}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cfg = _config(local=LocalConfig(default_model="qwen2.5-coder:7b"))
    runner = LocalRunner(cfg, client=client)

    res = asyncio.run(
        runner.run_subagent("problem_understander", "analyze this", system_prompt="you parse problems")
    )
    asyncio.run(client.aclose())

    assert isinstance(res, SubagentResult)
    assert res.backend == "local"
    assert res.model == "qwen2.5-coder:7b"
    assert '"ok": true' in res.text
    assert res.cost_usd == 0.0
    # request was OpenAI-compatible
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["content"] == "analyze this"
    assert captured["auth"] == "Bearer ollama"
    print("ok  test_local_runner_call_shape")


def test_local_runner_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model not found")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runner = LocalRunner(_config(local=LocalConfig()), client=client)
    try:
        asyncio.run(runner.run_subagent("problem_understander", "x"))
    except LocalBackendError:
        print("ok  test_local_runner_error")
    else:
        raise AssertionError("expected LocalBackendError on HTTP 500")
    finally:
        asyncio.run(client.aclose())


# --- router ---------------------------------------------------------------- #
class _FakeCloud:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_subagent(self, name, prompt, **kw):
        self.calls.append(name)
        return SubagentResult(name=name, model="claude", text="cloud", thinking="", backend="cloud")


class _FakeLocal:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def run_subagent(self, name, prompt, **kw):
        self.calls.append(name)
        if self.fail:
            raise LocalBackendError("unreachable")
        return SubagentResult(name=name, model="qwen", text="local", thinking="", backend="local")


def test_router_dispatch() -> None:
    cfg = _config(local=LocalConfig())
    cloud, local = _FakeCloud(), _FakeLocal()
    router = BackendRouter(cloud, local, cfg)

    # local-routed subagent goes local
    r1 = asyncio.run(router.run_subagent("problem_understander", "x"))
    assert r1.backend == "local" and local.calls == ["problem_understander"]
    # cloud-routed subagent goes cloud (and drops tool kwargs cleanly)
    r2 = asyncio.run(router.run_subagent("kernel_writer", "x", allowed_tools=["t"], mcp_servers={}))
    assert r2.backend == "cloud" and "kernel_writer" in cloud.calls
    print("ok  test_router_dispatch")


def test_router_falls_back_to_cloud_on_local_failure() -> None:
    cfg = _config(local=LocalConfig())
    cloud, local = _FakeCloud(), _FakeLocal(fail=True)
    notices: list[str] = []
    router = BackendRouter(cloud, local, cfg, on_notice=notices.append)

    # local fails -> cloud serves it, and local is disabled for the rest of the run
    r = asyncio.run(router.run_subagent("problem_understander", "x"))
    assert r.backend == "cloud"
    assert any("local backend unavailable" in n for n in notices)
    # second local-routed call skips local entirely now
    asyncio.run(router.run_subagent("library_updater", "x"))
    assert local.calls == ["problem_understander"]  # not retried
    assert cloud.calls == ["problem_understander", "library_updater"]
    print("ok  test_router_falls_back_to_cloud_on_local_failure")


if __name__ == "__main__":
    test_config_backend_for()
    test_local_runner_call_shape()
    test_local_runner_error()
    test_router_dispatch()
    test_router_falls_back_to_cloud_on_local_failure()
    print("\nall backend checks passed")
