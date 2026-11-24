"""Pick the model a request names, check what it supports, and run it on its engine adapter."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator

from catalog import ModelEntry
from errors import CapabilityError, EngineError, UnknownModel


class ModelRouter:
    """One loaded model per (model, quant); `parallel` generations run at once across all models."""

    def __init__(self, catalog: dict[str, ModelEntry], adapters: dict[str, Any], default: str,
                 quants: dict[str, str] | None = None, parallel: int = 1) -> None:
        if default not in catalog:
            raise UnknownModel(default)
        self.catalog, self.adapters, self.default = catalog, adapters, default
        self.quants = dict(quants or {})
        self._loaded: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        self.parallel = max(1, int(parallel))
        self._gate = asyncio.Semaphore(self.parallel)
        self.generation = 0  # part of the answer cache key; changes when loaded models are dropped

    @property
    def ready(self) -> bool:
        """The default model is loaded."""
        return (self.default, self.quant_of(self.catalog[self.default])) in self._loaded

    async def warm(self) -> None:
        """Load the default model now so the first request does not pay for it."""
        await asyncio.to_thread(self._model, self.catalog[self.default])

    async def close(self) -> None:
        self.forget()

    def entry(self, model: str | None) -> ModelEntry:
        try:
            return self.catalog[model or self.default]
        except KeyError:
            raise UnknownModel(model) from None

    def quant_of(self, entry: ModelEntry) -> str:
        return self.quants.get(entry.name, entry.default_quant)

    def set_quant(self, name: str, quant: str) -> None:
        entry = self.entry(name)
        entry.build(quant)  # KeyError for a quant the catalog does not list
        self.quants[name] = quant
        self.forget(name)

    def forget(self, name: str | None = None) -> None:
        with self._lock:
            for key in [key for key in self._loaded if name is None or key[0] == name]:
                del self._loaded[key]
            self.generation += 1

    def _model(self, entry: ModelEntry) -> tuple[Any, str]:
        quant = self.quant_of(entry)
        with self._lock:
            if (entry.name, quant) not in self._loaded:
                self._loaded[entry.name, quant] = self.adapters[entry.engine].load(entry, quant)
            return self._loaded[entry.name, quant], quant

    def _prepare(self, model: str | None, tools: list[dict[str, Any]] | None) -> tuple[ModelEntry, dict[str, Any]]:
        entry = self.entry(model)
        if tools and "tools" not in entry.capabilities:
            raise CapabilityError(f"{entry.name} does not support tools")
        return entry, ({"tools": tools} if tools else {})

    def _drop(self, entry: ModelEntry, error: EngineError) -> None:
        if error.status == 503:
            with self._lock:
                self._loaded.pop((entry.name, self.quant_of(entry)), None)

    async def chat(self, model: str | None, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int,
                   tools: list[dict[str, Any]] | None = None, timeout: float = 120.0) -> dict[str, Any]:
        entry, extra = self._prepare(model, tools)
        async with self._gate:
            try:
                loaded, quant = await asyncio.to_thread(self._model, entry)
                result = await asyncio.to_thread(loaded.generate, messages, temperature, max_tokens, timeout, **extra)
            except EngineError as error:
                self._drop(entry, error)
                raise
        return {**result, "model": entry.name, "quant": quant}

    async def chat_stream(self, model: str | None, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int,
                          tools: list[dict[str, Any]] | None = None, timeout: float = 120.0) -> AsyncIterator[tuple[str, Any]]:
        """("content", text) pieces, then ("done", stats with `model` and `quant`). Closing the iterator stops the generation."""
        entry, extra = self._prepare(model, tools)
        stop = threading.Event()
        async with self._gate:
            try:
                loaded, quant = await asyncio.to_thread(self._model, entry)
                events = loaded.stream(messages, temperature, max_tokens, timeout, stop, **extra)
                while (event := await asyncio.to_thread(next, events, None)) is not None:
                    kind, value = event
                    yield kind, ({**value, "model": entry.name, "quant": quant} if kind == "done" else value)
            except EngineError as error:
                self._drop(entry, error)
                raise
            finally:
                stop.set()
