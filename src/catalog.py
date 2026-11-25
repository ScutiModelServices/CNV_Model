"""Model catalog: one JSON file per model in `configs/models/`. See `configs/README.md`.

A model lists its builds (original or quantized). Each build lists the engines that can run it; which one runs is
chosen by the user, not by the catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CAPABILITIES = frozenset({"tools", "thinking", "schema", "project"})
ENGINES = frozenset({"ollama", "llamacpp"})
ENGINE_FIELDS = {"ollama": frozenset(), "llamacpp": frozenset({"ref", "url"})}  # what each engine needs from a build


class CatalogError(ValueError):
    """A catalog file is wrong. The message names the file."""


@dataclass(frozen=True)
class Build:
    quant: str
    file_mb: int  # stored size; the admin refresh replaces it with the live size when the Hub answers
    note: str = ""
    engines: dict[str, dict[str, str]] = field(default_factory=dict)  # engine -> its options for this build, in file order


@dataclass(frozen=True)
class ModelEntry:
    name: str
    title: str
    repo: str
    default_quant: str
    capabilities: frozenset[str]
    builds: tuple[Build, ...]

    def build(self, quant: str) -> Build:
        for item in self.builds:
            if item.quant == quant:
                return item
        raise KeyError(f"{self.name} has no build {quant}")

    def engines(self, quant: str) -> tuple[str, ...]:
        """The engines that can run this build; the first is the one used until the user picks another."""
        return tuple(self.build(quant).engines)

    def source(self, quant: str, engine: str) -> str:
        """What the engine is told to load: the Ollama model name, or `repo/file` for llama.cpp."""
        options = self.build(quant).engines[engine]
        return f"hf.co/{self.repo}:{quant}" if engine == "ollama" else f"{self.repo}/{options['ref']}"

    def url(self, quant: str, engine: str) -> str:
        """The sidecar that holds this build (llama.cpp); empty for engines that are one shared server."""
        return self.build(quant).engines[engine].get("url", "")


def _build(path: Path, raw: dict) -> Build:
    engines = raw.get("engines")
    if not isinstance(engines, dict) or not engines:
        raise CatalogError(f"{path.name}: build {raw.get('quant')} must list at least one engine")
    for engine, options in engines.items():
        if engine not in ENGINES:
            raise CatalogError(f"{path.name}: unknown engine '{engine}'")
        if not isinstance(options, dict) or set(options) != ENGINE_FIELDS[engine]:
            raise CatalogError(f"{path.name}: engine {engine} takes exactly {sorted(ENGINE_FIELDS[engine]) or 'no options'}")
    return Build(raw["quant"], int(raw["file_mb"]), raw.get("note", ""), {k: dict(v) for k, v in engines.items()})


def _entry(path: Path) -> ModelEntry:
    try:
        raw = json.loads(path.read_text("utf-8"))
        builds = tuple(_build(path, b) for b in raw["builds"])
        entry = ModelEntry(raw["name"], raw["title"], raw["repo"], raw["default_quant"],
                           frozenset(raw.get("capabilities", [])), builds)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CatalogError(f"{path.name}: {error}") from error
    if entry.name != path.stem:
        raise CatalogError(f"{path.name}: name '{entry.name}' must equal the file name")
    if not entry.capabilities <= CAPABILITIES:
        raise CatalogError(f"{path.name}: unknown capability {sorted(entry.capabilities - CAPABILITIES)}")
    if entry.default_quant not in {b.quant for b in builds}:
        raise CatalogError(f"{path.name}: default_quant is not one of the builds")
    return entry


def load_catalog(folder: Path) -> dict[str, ModelEntry]:
    entries = [_entry(path) for path in sorted(folder.glob("*.json"))]
    if not entries:
        raise CatalogError(f"no model files in {folder}")
    return {entry.name: entry for entry in entries}
