"""Ollama adapter: chat through /api/chat, weights through /api/pull; blocking calls run in router worker threads."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from http.client import HTTPException
from typing import Any, Callable, Iterator
from urllib.parse import quote, urlparse

from catalog import ModelEntry
from errors import EngineError

DEFAULT_URL = "http://127.0.0.1:11434"


def _call(url: str, path: str, body: dict[str, Any] | None, timeout: float):
    request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code == 400:
            raise EngineError("Ollama refused the request (400)", 400) from None
        model = (body or {}).get("model", "the configured model")
        raise EngineError("Model unavailable; start Ollama and ask an admin to press Download on the Updates card.", 503) from None
    except OSError:
        model = (body or {}).get("model", "the configured model")
        raise EngineError("Model unavailable; start Ollama and ask an admin to press Download on the Updates card.", 503) from None


_READ_ERRORS = (OSError, HTTPException, ValueError, TypeError, KeyError, AttributeError)


def _unavailable(model: str) -> EngineError:
    return EngineError("Model unavailable; start Ollama and ask an admin to press Download on the Updates card.", 503)


def _json_call(url: str, path: str, body: dict[str, Any] | None, timeout: float, model: str) -> dict[str, Any]:
    """Read and validate a backend object without exposing response bodies or transport details."""
    try:
        with _call(url, path, body, timeout) as response:
            reply = json.load(response)
        if not isinstance(reply, dict) or reply.get("error"):
            raise _unavailable(model)
        return reply
    except _READ_ERRORS:
        raise _unavailable(model) from None


class OllamaLoaded:
    """Answer generations through Ollama's /api/chat."""

    def __init__(self, url: str, model: str, load_timeout: float = 300.0, thinking: bool = False,
                 hide_reasoning: bool = True, live: Callable[[], tuple[bool, bool]] | None = None) -> None:
        """Read live thinking/visibility settings for each generation."""
        self.url, self.model, self._fixed, self._live = url.rstrip("/"), model, (thinking, hide_reasoning), live
        self.num_thread: int | None = None  # CPU threads Ollama uses per request; None = Ollama decides
        _json_call(self.url, "/api/generate", {"model": model, "prompt": "", "stream": False}, load_timeout, model)

    @property
    def thinking(self) -> bool:
        return (self._live or (lambda: self._fixed))()[0]

    @property
    def hide_reasoning(self) -> bool:
        return (self._live or (lambda: self._fixed))()[1]

    def _body(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, stream: bool,
              tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "stream": stream, "think": self.thinking,
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if self.num_thread:
            body["options"]["num_thread"] = self.num_thread
        if tools:
            body["tools"] = tools
        return body

    def _clean(self, text: str, hide: bool | None = None) -> str:
        """Drop reasoning written into content through its final closing marker."""
        return text.rsplit("</think>", 1)[-1].strip() if (self.hide_reasoning if hide is None else hide) and "</think>" in text else text

    @staticmethod
    def _stats(last: dict[str, Any]) -> dict[str, Any]:
        return {"prompt_tokens": last.get("prompt_eval_count", 0), "completion_tokens": last.get("eval_count", 0),
                "finish": "length" if last.get("done_reason") == "length" else "stop"}

    def stream(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, max_time: float,
               stop: Any, tools: list[dict[str, Any]] | None = None) -> Iterator[tuple[str, Any]]:
        hide = self.hide_reasoning
        held: list[str] = []
        try:
            with _call(self.url, "/api/chat", self._body(messages, temperature, max_tokens, True, tools), max_time) as response:
                for raw in response:
                    if stop.is_set():
                        return
                    piece = json.loads(raw)
                    if not isinstance(piece, dict) or piece.get("error"):
                        raise _unavailable(self.model)
                    message = piece.get("message", {})
                    if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
                        raise _unavailable(self.model)
                    if text := message.get("content"):
                        if hide:
                            held.append(text)
                        else:
                            yield "content", text
                    if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
                        yield "tool_calls", message["tool_calls"]
                    if piece.get("done") is True:
                        if held and (answer := self._clean("".join(held), hide)):
                            yield "content", answer
                        yield "done", self._stats(piece)
                        return
                if not stop.is_set():
                    raise _unavailable(self.model)
        except _READ_ERRORS:
            raise _unavailable(self.model) from None

    def generate(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, max_time: float,
                 tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        reply = _json_call(self.url, "/api/chat", self._body(messages, temperature, max_tokens, False, tools), max_time, self.model)
        try:
            content = reply["message"]["content"]
            if not isinstance(content, str):
                raise _unavailable(self.model)
            calls = reply["message"].get("tool_calls") or []
            if not isinstance(calls, list):
                raise _unavailable(self.model)
            return {"content": self._clean(content.strip()), **({"tool_calls": calls} if calls else {}), **self._stats(reply)}
        except _READ_ERRORS:
            raise _unavailable(self.model) from None

class OllamaAdapter:
    """One Ollama server for every `engine: ollama` model. `live()` returns (thinking, hide_reasoning) per generation."""

    name = "ollama"

    def __init__(self, url: str = DEFAULT_URL, live: Callable[[], tuple[bool, bool]] | None = None) -> None:
        self.url = url.strip().rstrip("/")  # changed live by the admin setting `ollama_url`
        if not self.url:
            raise ValueError("The Ollama server address must not be empty")
        self._live = live

    @property
    def same_machine(self) -> bool:
        return urlparse(self.url).hostname in ("127.0.0.1", "localhost", "::1")

    def load(self, entry: ModelEntry, quant: str) -> OllamaLoaded:
        return OllamaLoaded(self.url, entry.source(quant, "ollama"), live=self._live)

    def unload(self, entry: ModelEntry, quant: str) -> None:
        """Ask Ollama to free the model now (keep_alive 0)."""
        model = entry.source(quant, "ollama")
        _json_call(self.url, "/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": 0}, 30, model)

    def delete(self, entry: ModelEntry, quant: str) -> None:
        """Remove the model from Ollama's storage; already gone is fine."""
        model = entry.source(quant, "ollama")
        request = urllib.request.Request(f"{self.url}/api/delete", data=json.dumps({"model": model}).encode(), method="DELETE",
                                         headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=60).close()
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise _unavailable(model) from None
        except OSError:
            raise _unavailable(model) from None

    def downloaded(self) -> set[str]:
        """Names of every model Ollama has on disk, with and without an implicit `:latest`."""
        reply = _json_call(self.url, "/api/tags", None, 15, "")
        try:
            return {name.removesuffix(":latest") if name.endswith(":latest") and name.count(":") == 1 else name
                    for name in (m["name"] for m in reply["models"])}
        except _READ_ERRORS:
            raise _unavailable("") from None

    def download(self, entry: ModelEntry, quant: str) -> None:
        from progress import ollama_pull_progress

        model = entry.source(quant, "ollama")
        with _call(self.url, "/api/pull", {"model": model, "stream": True}, 3600) as response:
            status = ollama_pull_progress(response, model)
        if status != "success":
            raise _unavailable(model)

    def local_ref(self, entry: ModelEntry, quant: str) -> str | None:
        model = entry.source(quant, "ollama")
        reply = _json_call(self.url, "/api/tags", None, 15, model)
        try:
            digest = next((m["digest"] for m in reply["models"] if m["name"] in (model, f"{model}:latest")), None)
            return digest.removeprefix("sha256:").lower() if digest else None
        except _READ_ERRORS:
            raise _unavailable(model) from None

    def remote_ref(self, entry: ModelEntry, quant: str) -> str:
        """Hash the registry manifest bytes, matching Ollama's installed digest."""
        name, host = entry.source(quant, "ollama"), "registry.ollama.ai"
        if name.startswith(("hf.co/", "huggingface.co/")):
            host, name = name.split("/", 1)
        name, separator, tag = name.partition(":")
        if not separator:
            tag = "latest"
        if host == "registry.ollama.ai" and "/" not in name:
            name = "library/" + name
        request = urllib.request.Request(
            f"https://{host}/v2/{quote(name, safe='/')}/manifests/{quote(tag, safe='')}",
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
        manifest = json.loads(raw)
        if (not isinstance(manifest, dict) or manifest.get("schemaVersion") != 2
                or not isinstance(manifest.get("config"), dict) or not isinstance(manifest.get("layers"), list)):
            raise EngineError("Registry returned an invalid model manifest", 503)
        return hashlib.sha256(raw).hexdigest()
