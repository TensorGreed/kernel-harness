"""Local-model backend + per-subagent router.

Complements the cloud Claude backend (``AgentRunner``) with a local
OpenAI-compatible one (``LocalRunner`` → Ollama / vLLM on the GB10), so the
mechanical, tool-free subagents can run for free on-device while the tool-using,
highest-stakes roles stay on Claude.

The integration is seamless because :class:`BackendRouter` implements the *same*
``run_subagent(name, prompt, ...)`` interface the subagents already call — the
orchestrator just hands the router in as the "runner", and nothing in
``subagents.py`` changes. The router dispatches per subagent via
``RunConfig.backend_for`` and **falls back to cloud per-call** if the local
endpoint errors or is unreachable, so "local by default" never breaks a run when
Ollama isn't running. See CONTEXT.md → "Possible extension".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from .agent import AgentRunner, SubagentResult
from .config import RunConfig


class LocalBackendError(RuntimeError):
    """The local endpoint failed (unreachable, bad response, model missing)."""


class LocalRunner:
    """Calls an OpenAI-compatible chat endpoint (Ollama / vLLM).

    Tool-free by design — only ``system_prompt`` + ``prompt`` are used; any tool
    kwargs are ignored (the router only routes tool-free subagents here).
    """

    def __init__(self, config: RunConfig, *, client: httpx.AsyncClient | None = None) -> None:
        if config.local is None:
            raise LocalBackendError("LocalRunner requires config.local to be set")
        self.config = config
        self.local = config.local
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.local.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def run_subagent(
        self,
        name: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        on_text: Callable[[str], None] | None = None,
        **_ignored: Any,
    ) -> SubagentResult:
        model = self.local.model_for(name)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        url = self.local.base_url.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "stream": False}
        headers = {"Authorization": f"Bearer {self.local.api_key}"}

        client = await self._get_client()
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LocalBackendError(f"local backend call failed for {name!r}: {exc}") from exc

        text = _extract_content(data)
        if on_text is not None and text:
            on_text(text)
        usage = data.get("usage") if isinstance(data, dict) else None
        return SubagentResult(
            name=name, model=model, text=text, thinking="",
            usage=usage, cost_usd=0.0, backend="local",
        )


def _extract_content(data: dict) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LocalBackendError(f"unexpected response shape: {exc}") from exc


class BackendRouter:
    """Dispatches each subagent to cloud or local, with cloud fallback.

    Implements the ``run_subagent`` interface so it is a drop-in "runner" for the
    subagents. Tool-using subagents must be routed to cloud (the local backend
    has no tool loop); the default routing already does this.
    """

    def __init__(
        self,
        cloud: AgentRunner,
        local: LocalRunner | None,
        config: RunConfig,
        *,
        on_notice: Callable[[str], None] | None = None,
    ) -> None:
        self.cloud = cloud
        self.local = local
        self.config = config
        self._on_notice = on_notice or (lambda _m: None)
        self._local_disabled = False  # set after a failure to avoid repeated retries

    async def run_subagent(self, name: str, prompt: str, **kwargs: Any) -> SubagentResult:
        wants_local = (
            self.config.backend_for(name) == "local"
            and self.local is not None
            and not self._local_disabled
        )
        if wants_local:
            try:
                return await self.local.run_subagent(name, prompt, **kwargs)
            except LocalBackendError as exc:
                # Fall back to cloud for this call; disable local for the rest of
                # the run so we don't keep paying the timeout on every call.
                self._local_disabled = True
                self._on_notice(
                    f"local backend unavailable ({exc}); using cloud for {name!r} "
                    "and the rest of this run."
                )
        return await self.cloud.run_subagent(name, prompt, **kwargs)
