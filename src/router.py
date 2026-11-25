"""Pick the model a request names, check what it supports, and run it on the engine the user chose."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, Callable

from catalog import ModelEntry
from errors import CapabilityError, EngineError, ModelStopped, UnknownModel


class ModelRouter:
    """One loaded model per (model, build, engine); `parallel` generations run at once across all models."""

    def __init__(self, catalog: dict[str, ModelEntry], adapters: dict[str, Any],
                 quants: dict[str, str] | None = None, parallel: int = 1, engines: dict[str, str] | None = None,
                 guard: Callable[[ModelEntry, str, str], str | None] | None = None) -> None:
        self.catalog, self.adapters = catalog, adapters
        self.stopped: set[str] = set()  # models the user stopped: requests are refused until they are started
        self.guard = guard  # (model, build, engine) -> why it cannot be loaded here, or None
        self.quants, self.engines = dict(quants or {}), dict(engines or {})  # the user's choice per model
        self._loaded: dict[tuple[str, str, str], Any] = {}
        self._lock = threading.Lock()
        self.parallel = max(1, int(parallel))
        self._gate = asyncio.Semaphore(self.parallel)
        self.generation = 0  # part of the answer cache key; changes when loaded models are dropped

    @property
    def ready(self) -> bool:
        """Some model is loaded."""
        return bool(self._loaded)

    async def close(self) -> None:
        self.forget()

    def set_parallel(self, count: int) -> None:
        """Generations still running keep the old gate; new ones use the new one."""
        self.parallel = max(1, int(count))
        self._gate = asyncio.Semaphore(self.parallel)

    def entry(self, model: str) -> ModelEntry:
        try:
            return self.catalog[model]
        except KeyError:
            raise UnknownModel(model) from None

    def quant_of(self, entry: ModelEntry) -> str | None:
        """The build the user selected; None until they select one (there is no default)."""
        return self.quants.get(entry.name)

    def engine_of(self, entry: ModelEntry, engine: str | None = None) -> str:
        """The engine for this model: the request's, else the user's saved choice, else the first the build lists.
        CapabilityError when the build cannot run on it."""
        if self.quant_of(entry) is None:
            raise EngineError(f"{entry.name} has no selected build: download one and select it on the admin page.", 503)
        listed = entry.engines(self.quant_of(entry))
        chosen = engine or self.engines.get(entry.name) or listed[0]
        if chosen not in listed:
            raise CapabilityError(f"{entry.name} {self.quant_of(entry)} cannot run on {chosen}; it runs on: {', '.join(listed)}")
        return chosen

    def set_choice(self, name: str, quant: str, engine: str) -> None:
        """Run `name` as this build on this engine from now on."""
        entry = self.entry(name)
        if engine not in entry.engines(quant):  # KeyError for a build the catalog does not list
            raise CapabilityError(f"{name} {quant} cannot run on {engine}")
        self.quants[name], self.engines[name] = quant, engine
        self.forget(name)

    def state_of(self, name: str) -> str:
        """`stopped` (refuses requests), `running` (loaded here) or `ready` (loads on its first request)."""
        if name in self.stopped:
            return "stopped"
        return "running" if any(key[0] == name for key in self._loaded) else "ready"

    async def start(self, name: str) -> None:
        """Allow requests again and load the model now."""
        entry = self.entry(name)
        self.stopped.discard(name)
        await asyncio.to_thread(self._model, entry, self.engine_of(entry))

    async def stop(self, name: str) -> None:
        """Refuse requests and free the model: drop the handle here and tell the engine to unload it."""
        entry = self.entry(name)
        self.stopped.add(name)
        self.forget(name)
        try:
            await asyncio.to_thread(self.adapters[self.engine_of(entry)].unload, entry, self.quant_of(entry))
        except EngineError:
            pass  # the engine is down or does not have it: nothing is loaded

    def forget(self, name: str | None = None) -> None:
        with self._lock:
            for key in [key for key in self._loaded if name is None or key[0] == name]:
                del self._loaded[key]
            self.generation += 1

    def _model(self, entry: ModelEntry, engine: str) -> tuple[Any, str]:
        quant = self.quant_of(entry)
        with self._lock:
            if (entry.name, quant, engine) not in self._loaded:
                if self.guard and (reason := self.guard(entry, quant, engine)):
                    raise EngineError(reason, 503)
                self._loaded[entry.name, quant, engine] = self.adapters[engine].load(entry, quant)
            return self._loaded[entry.name, quant, engine], quant

    def _prepare(self, model: str, engine: str | None, tools: list[dict[str, Any]] | None) -> tuple[ModelEntry, str, dict[str, Any]]:
        entry = self.entry(model)
        if entry.name in self.stopped:
            raise ModelStopped(entry.name)
        engine = self.engine_of(entry, engine)
        if tools and "tools" not in entry.capabilities:
            raise CapabilityError(f"{entry.name} does not support tools")
        return entry, engine, ({"tools": tools} if tools else {})

    def _drop(self, entry: ModelEntry, engine: str, error: EngineError) -> None:
        if error.status == 503:
            with self._lock:
                self._loaded.pop((entry.name, self.quant_of(entry), engine), None)

    async def chat(self, model: str, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int,
                   tools: list[dict[str, Any]] | None = None, engine: str | None = None, timeout: float = 120.0) -> dict[str, Any]:
        entry, engine, extra = self._prepare(model, engine, tools)
        async with self._gate:
            try:
                loaded, quant = await asyncio.to_thread(self._model, entry, engine)
                result = await asyncio.to_thread(loaded.generate, messages, temperature, max_tokens, timeout, **extra)
            except EngineError as error:
                self._drop(entry, engine, error)
                raise
        return {**result, "model": entry.name, "quant": quant, "engine": engine}

    async def chat_stream(self, model: str, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int,
                          tools: list[dict[str, Any]] | None = None, engine: str | None = None,
                          timeout: float = 120.0) -> AsyncIterator[tuple[str, Any]]:
        """("content", text) pieces, then ("done", stats with `model`, `quant` and `engine`). Closing the iterator stops the generation."""
        entry, engine, extra = self._prepare(model, engine, tools)
        stop = threading.Event()
        async with self._gate:
            try:
                loaded, quant = await asyncio.to_thread(self._model, entry, engine)
                events = loaded.stream(messages, temperature, max_tokens, timeout, stop, **extra)
                while (event := await asyncio.to_thread(next, events, None)) is not None:
                    kind, value = event
                    yield kind, ({**value, "model": entry.name, "quant": quant, "engine": engine} if kind == "done" else value)
            except EngineError as error:
                self._drop(entry, engine, error)
                raise
            finally:
                stop.set()
