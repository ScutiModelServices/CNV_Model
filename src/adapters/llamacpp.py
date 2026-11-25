"""llama.cpp adapter: a `llama-server` sidecar answers OpenAI-style chat; weights are GGUF files in a shared folder.

A sidecar loads one GGUF (`--model <models_dir>/<repo>/<ref>`), so each `llamacpp` model names its own server in the
catalog (`url`)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from catalog import ModelEntry
from errors import EngineError

UNAVAILABLE = "Model unavailable; start the llama.cpp server and ask an admin to press Download on the model's build."
HUB = "https://huggingface.co"


CONTEXT = 4096  # the bitnet.cpp server is started with -c 4096
END_OF_TURN = "<|eot_id|>"


def render_bitnet(messages: list[dict[str, Any]]) -> str:
    """`Role: text<|eot_id|>` per message, then `Assistant: ` (the template of microsoft/bitnet-b1.58-2B-4T). The GGUF
    carries no chat template and the bitnet.cpp server cannot read one, so the service writes the prompt itself."""
    return "".join(f"{m['role'].capitalize()}: {m['content'].strip()}{END_OF_TURN}" for m in messages) + "Assistant: "


class _Loaded:
    """One sidecar. `prompt` None: the server has a chat template and answers /v1/chat/completions. `prompt` "bitnet":
    the service renders the prompt and uses /completion."""

    def __init__(self, adapter: "LlamaCppAdapter", model: str, url: str, prompt: str | None = None) -> None:
        self._adapter, self._model, self._url, self._prompt = adapter, model, url, prompt

    def _open(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, max_time: float, stream: bool):
        send = self._adapter._open
        if self._prompt != "bitnet":
            return send(self._url, "/v1/chat/completions", {"model": self._model, "messages": messages, "temperature": temperature,
                                                            "max_tokens": max_tokens, "stream": stream}, max_time)
        prompt = render_bitnet(messages)
        with send(self._url, "/tokenize", {"content": prompt}, max_time) as reply:
            size = len(json.load(reply)["tokens"])
        if size >= CONTEXT:  # the server would cut it silently
            raise EngineError(f"The conversation is {size} tokens; the model reads at most {CONTEXT}", 400)
        # the server runs with --special, so <|eot_id|> is printed and can end the answer
        return send(self._url, "/completion", {"prompt": prompt, "temperature": temperature, "n_predict": min(max_tokens, CONTEXT - size),
                                               "stream": stream, "stop": [END_OF_TURN]}, max_time)

    def generate(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, max_time: float) -> dict[str, Any]:
        try:
            with self._open(messages, temperature, max_tokens, max_time, False) as response:
                reply = json.load(response)
            if self._prompt == "bitnet":
                return {"content": reply["content"].strip(), "prompt_tokens": reply.get("tokens_evaluated", 0),
                        "completion_tokens": reply.get("tokens_predicted", 0), "finish": "length" if reply.get("stopped_limit") else "stop"}
            choice, usage = reply["choices"][0], reply.get("usage", {})
            return {"content": choice["message"]["content"].strip(), "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "finish": "length" if choice.get("finish_reason") == "length" else "stop"}
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise EngineError(UNAVAILABLE, 503) from None

    def stream(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int, max_time: float, stop: Any) -> Iterator[tuple[str, Any]]:
        finish, stats = "stop", {}
        try:
            with self._open(messages, temperature, max_tokens, max_time, True) as response:
                for raw in response:
                    if stop.is_set():
                        return
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    piece = json.loads(line[5:])
                    if self._prompt == "bitnet":
                        if text := piece.get("content"):
                            yield "content", text
                        if piece.get("stop"):
                            stats = {"prompt_tokens": piece.get("tokens_evaluated", 0), "completion_tokens": piece.get("tokens_predicted", 0)}
                            finish = "length" if piece.get("stopped_limit") else "stop"
                        continue
                    stats = piece.get("usage") or stats
                    choice = piece["choices"][0]
                    if text := choice.get("delta", {}).get("content"):
                        yield "content", text
                    if choice.get("finish_reason"):
                        finish = "length" if choice["finish_reason"] == "length" else "stop"
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            raise EngineError(UNAVAILABLE, 503) from None
        yield "done", {"prompt_tokens": stats.get("prompt_tokens", 0), "completion_tokens": stats.get("completion_tokens", 0), "finish": finish}


class LlamaCppAdapter:
    """The `llama-server` sidecars of the `engine: llamacpp` models; GGUF files live in one folder shared with them."""

    name = "llamacpp"

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self.same_machine = True  # the sidecars share this folder, so they run beside this service

    def _open(self, url: str, path: str, body: dict[str, Any], timeout: float):
        request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            raise EngineError("llama.cpp refused the request (400)" if error.code == 400 else UNAVAILABLE, 400 if error.code == 400 else 503) from None
        except OSError:
            raise EngineError(UNAVAILABLE, 503) from None

    def _post(self, url: str, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            with self._open(url, path, body, timeout) as response:
                reply = json.load(response)
        except (OSError, ValueError):
            raise EngineError(UNAVAILABLE, 503) from None
        if not isinstance(reply, dict) or reply.get("error"):
            raise EngineError(UNAVAILABLE, 503)
        return reply

    def load(self, entry: ModelEntry, quant: str) -> _Loaded:
        if not (self.models_dir / entry.source(quant, "llamacpp")).is_file():
            raise EngineError(UNAVAILABLE, 503)
        return _Loaded(self, entry.source(quant, "llamacpp"), entry.url(quant, "llamacpp"), entry.options(quant, "llamacpp").get("prompt"))

    def unload(self, entry: ModelEntry, quant: str) -> None:
        """Nothing to free: the sidecar keeps its one model in memory for as long as it runs."""

    def delete(self, entry: ModelEntry, quant: str) -> None:
        """Remove the GGUF file (and a half-finished download). A running sidecar keeps its open copy until it restarts."""
        target = self.models_dir / entry.source(quant, "llamacpp")
        try:
            target.unlink(missing_ok=True)
            target.with_name(target.name + ".part").unlink(missing_ok=True)
        except OSError:
            raise EngineError("Could not delete the file.", 500) from None

    def downloaded(self) -> set[str]:
        return {path.relative_to(self.models_dir).as_posix() for path in self.models_dir.rglob("*.gguf")} if self.models_dir.is_dir() else set()

    def download(self, entry: ModelEntry, quant: str) -> None:
        from progress import progress  # shared download progress, same one the Ollama pull reports to

        target = self.models_dir / entry.source(quant, "llamacpp")
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        request = urllib.request.Request(f"{HUB}/{entry.repo}/resolve/main/{entry.build(quant).engines["llamacpp"]["ref"]}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                done = 0
                progress.start(entry.source(quant, "llamacpp"), int(response.headers.get("Content-Length") or 0) or None)
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
        path = self.models_dir / entry.source(quant, "llamacpp")
        return str(path.stat().st_size) if path.is_file() else None

    def remote_ref(self, entry: ModelEntry, quant: str) -> str:
        request = urllib.request.Request(f"{HUB}/{entry.repo}/resolve/main/{entry.build(quant).engines["llamacpp"]["ref"]}", method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.headers.get("Content-Length", "")
        except OSError:
            return ""
