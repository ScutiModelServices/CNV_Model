"""llama.cpp adapter: a `llama-server` sidecar answers OpenAI-style chat; weights are GGUF files in a shared folder.

The sidecar loads one GGUF (`--model <models_dir>/<repo>/<ref>`), so one URL serves the one model that points at it.
shortcut: one sidecar URL for all `llamacpp` models; add a url field to the catalog when a second one is needed."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from catalog import ModelEntry
from errors import EngineError

UNAVAILABLE = "Model unavailable; start the llama.cpp server and ask an admin to press Download on the model's build."
HUB = "https://huggingface.co"


class _Loaded:
    def __init__(self, adapter: "LlamaCppAdapter", model: str) -> None:
        self._adapter, self._model = adapter, model

    def _body(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, stream: bool) -> dict[str, Any]:
        return {"model": self._model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": stream}

    def generate(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, max_time: float) -> dict[str, Any]:
        reply = self._adapter._post("/v1/chat/completions", self._body(messages, temperature, max_tokens, False), max_time)
        try:
            choice, usage = reply["choices"][0], reply.get("usage", {})
            return {"content": choice["message"]["content"].strip(), "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "finish": "length" if choice.get("finish_reason") == "length" else "stop"}
        except (KeyError, IndexError, TypeError, AttributeError):
            raise EngineError(UNAVAILABLE, 503) from None

    def stream(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, max_time: float, stop: Any) -> Iterator[tuple[str, Any]]:
        finish, usage = "stop", {}
        try:
            with self._adapter._open("/v1/chat/completions", self._body(messages, temperature, max_tokens, True), max_time) as response:
                for raw in response:
                    if stop.is_set():
                        return
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    piece = json.loads(data)
                    usage = piece.get("usage") or usage
                    choice = piece["choices"][0]
                    if text := choice.get("delta", {}).get("content"):
                        yield "content", text
                    if choice.get("finish_reason"):
                        finish = "length" if choice["finish_reason"] == "length" else "stop"
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            raise EngineError(UNAVAILABLE, 503) from None
        yield "done", {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "finish": finish}


class LlamaCppAdapter:
    """One `llama-server` for every `engine: llamacpp` model."""

    name = "llamacpp"

    def __init__(self, url: str, models_dir: Path) -> None:
        self.url, self.models_dir = url.strip().rstrip("/"), Path(models_dir)
        self.same_machine = urlparse(self.url).hostname in ("127.0.0.1", "localhost", "::1", "host.docker.internal")

    def _open(self, path: str, body: dict[str, Any], timeout: float):
        request = urllib.request.Request(f"{self.url}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            raise EngineError("llama.cpp refused the request (400)" if error.code == 400 else UNAVAILABLE, 400 if error.code == 400 else 503) from None
        except OSError:
            raise EngineError(UNAVAILABLE, 503) from None

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            with self._open(path, body, timeout) as response:
                reply = json.load(response)
        except (OSError, ValueError):
            raise EngineError(UNAVAILABLE, 503) from None
        if not isinstance(reply, dict) or reply.get("error"):
            raise EngineError(UNAVAILABLE, 503)
        return reply

    def load(self, entry: ModelEntry, quant: str) -> _Loaded:
        if not (self.models_dir / entry.source(quant)).is_file():
            raise EngineError(UNAVAILABLE, 503)
        return _Loaded(self, entry.source(quant))

    def downloaded(self) -> set[str]:
        return {path.relative_to(self.models_dir).as_posix() for path in self.models_dir.rglob("*.gguf")} if self.models_dir.is_dir() else set()

    def download(self, entry: ModelEntry, quant: str) -> None:
        from progress import progress  # shared download progress, same one the Ollama pull reports to

        target = self.models_dir / entry.source(quant)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        request = urllib.request.Request(f"{HUB}/{entry.repo}/resolve/main/{entry.build(quant).ref}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                done = 0
                progress.start(entry.source(quant), int(response.headers.get("Content-Length") or 0) or None)
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    progress.set(done)
        except OSError:
            part.unlink(missing_ok=True)
            raise EngineError(UNAVAILABLE, 503) from None
        finally:
            progress.finish()
        part.replace(target)

    def local_ref(self, entry: ModelEntry, quant: str) -> str | None:
        path = self.models_dir / entry.source(quant)
        return str(path.stat().st_size) if path.is_file() else None

    def remote_ref(self, entry: ModelEntry, quant: str) -> str:
        request = urllib.request.Request(f"{HUB}/{entry.repo}/resolve/main/{entry.build(quant).ref}", method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.headers.get("Content-Length", "")
        except OSError:
            return ""
