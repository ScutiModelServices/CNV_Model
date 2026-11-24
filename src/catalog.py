"""Model catalog: one JSON file per model in `configs/models/`. See `configs/README.md`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CAPABILITIES = frozenset({"tools", "thinking", "schema", "project"})
ENGINES = frozenset({"ollama", "llamacpp"})


class CatalogError(ValueError):
    """A catalog file is wrong. The message names the file."""


@dataclass(frozen=True)
class Build:
    quant: str
    file_mb: int  # stored size; the admin refresh replaces it with the live size when the Hub answers
    note: str = ""
    ref: str = ""  # llamacpp: GGUF file name inside `repo`; ollama: unused (the tag is the quant)


@dataclass(frozen=True)
class ModelEntry:
    name: str
    title: str
    engine: str
    repo: str
    default_quant: str
    capabilities: frozenset[str]
    builds: tuple[Build, ...]
    url: str = ""  # llamacpp only: the llama-server that holds this model's GGUF

    def build(self, quant: str) -> Build:
        for item in self.builds:
            if item.quant == quant:
                return item
        raise KeyError(f"{self.name} has no build {quant}")

    def source(self, quant: str) -> str:
        """What the engine is told to load: the Ollama model name, or `repo/file` for llama.cpp."""
        build = self.build(quant)
        return f"hf.co/{self.repo}:{quant}" if self.engine == "ollama" else f"{self.repo}/{build.ref}"


def _entry(path: Path) -> ModelEntry:
    try:
        raw = json.loads(path.read_text("utf-8"))
        builds = tuple(Build(b["quant"], int(b["file_mb"]), b.get("note", ""), b.get("ref", "")) for b in raw["builds"])
        entry = ModelEntry(raw["name"], raw["title"], raw["engine"], raw["repo"], raw["default_quant"],
                           frozenset(raw.get("capabilities", [])), builds, raw.get("url", ""))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CatalogError(f"{path.name}: {error}") from error
    if entry.name != path.stem:
        raise CatalogError(f"{path.name}: name '{entry.name}' must equal the file name")
    if entry.engine not in ENGINES:
        raise CatalogError(f"{path.name}: unknown engine '{entry.engine}'")
    if (entry.engine == "llamacpp") != bool(entry.url):
        raise CatalogError(f"{path.name}: `url` is required for engine llamacpp and only allowed there")
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
