"""Rows for the admin page's build list under each model, and the live file sizes behind them.

The memory estimate is arithmetic, not a measurement: file size * 1.1 + 400 MB for context and runtime.
shortcut: free memory is read when the page is opened and does not know what the engine loads next."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from catalog import ModelEntry

FREE_SHARE = {"gpu": 0.9, "cpu": 0.85}
UNKNOWN = "Free memory could not be read."
REMOTE = "The engine runs on another machine, so its memory cannot be checked."
_QUANT = re.compile(r"[-_.]((?:IQ|Q)\d\w*|F16|BF16|F32|i2_s)\.gguf$", re.IGNORECASE)


def cpu_flags() -> set[str] | None:
    """CPU feature flags of this machine (Linux, also inside a container: it shows the host's CPU); None when unreadable."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("flags", "Features")):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return None


def missing_cpu(entry: ModelEntry, quant: str, engine: str, flags: set[str] | None) -> str | None:
    """Why the CPU cannot run this build (it lacks a required flag), or None (fine, or the flags cannot be read)."""
    needs = entry.options(quant, engine).get("cpu")
    if needs and flags is not None and needs not in flags:
        return f"Needs a CPU with {needs.upper()}; this one does not have it."
    return None


def estimate_mb(file_mb: int) -> int:
    return round(file_mb * 1.1 + 400)


def place(need_mb: int, hardware: dict[str, Any]) -> tuple[str | None, str]:
    gpu, free = hardware.get("gpu"), hardware.get("ram_available_mb")
    if gpu and need_mb <= gpu["vram_free_mb"] * FREE_SHARE["gpu"]:
        return "gpu", ""
    if free is None:
        return None, UNKNOWN
    if need_mb <= free * FREE_SHARE["cpu"]:
        return "cpu", ""
    gpu_text = f" or {gpu['vram_free_mb']} MB of GPU memory" if gpu else ""
    return None, f"Needs about {need_mb} MB; {free} MB of memory{gpu_text} is free."


def listing(entry: ModelEntry, hardware: dict[str, dict[str, Any] | None], current: tuple[str, str],
            downloaded: dict[str, set[str]], live: dict[str, int], flags: set[str] | None = None) -> list[dict[str, Any]]:
    """Every build of `entry`, smallest first, each with the engines that can run it. `hardware[engine]` is what that
    engine's machine has (None when it is another machine); `downloaded[engine]` the sources it has on disk;
    `current` is the (build, engine) the user runs now."""
    rows = []
    for build in entry.builds:
        mb = live.get(build.quant, build.file_mb)
        need = estimate_mb(mb)
        options = []
        for engine in build.engines:
            in_use = (build.quant, engine) == current
            hw = hardware.get(engine)
            where, reason = place(need, hw) if hw is not None else (None, REMOTE)
            fits = True if in_use or where else (None if reason in (UNKNOWN, REMOTE) else False)
            cpu = missing_cpu(entry, build.quant, engine, flags)  # even the running build: it cannot work
            if cpu:
                fits, reason, where = False, cpu, None
            options.append({"engine": engine, "current": in_use, "fits": fits, "reason": reason if cpu else "" if in_use or fits else reason,
                            "runs_on": where, "source": entry.source(build.quant, engine),
                            "downloaded": entry.source(build.quant, engine) in downloaded.get(engine, set())})
        rows.append({"id": build.quant, "title": build.quant, "quant": build.quant, "note": build.note,
                     "detail": f"about {need} MB · {mb} MB download · {build.note}",
                     "disk_mb": mb, "estimated_mb": need, "current": build.quant == current[0], "engines": options})
    return sorted(rows, key=lambda row: row["estimated_mb"])


def _quant_key(text: str) -> str:
    return text if text[0] == "i" else text.upper()


def hub_files(repo: str, fetch: Callable[[str], list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Quant -> {"path", "mb"} for the single-file GGUF builds of a Hub repo. `fetch(url)` returns the file-tree JSON."""
    files = {}
    for item in fetch(f"https://huggingface.co/api/models/{repo}/tree/main"):
        match = _QUANT.search(item.get("path", ""))
        if match and item.get("size"):
            files[_quant_key(match.group(1))] = {"path": item["path"], "mb": round(item["size"] / 1_048_576)}
    return files


def hub_sizes(repo: str, fetch: Callable[[str], list[dict[str, Any]]]) -> dict[str, int]:
    """Quant -> size in MB for the GGUF files of a Hub repo."""
    return {quant: item["mb"] for quant, item in hub_files(repo, fetch).items()}


def cached_files(path: Path, repo: str, fetch: Callable[[str], list[dict[str, Any]]], now: float, ttl: float = 86400) -> dict[str, dict[str, Any]]:
    """Hub files with a JSON cache in the data volume. A fresh cache skips the Hub; a Hub failure keeps the old cache."""
    try:
        cache = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(repo)
    if hit and now - hit["at"] < ttl:
        return hit["files"]
    try:
        files = hub_files(repo, fetch)
    except (OSError, ValueError):
        return hit["files"] if hit else {}
    cache[repo] = {"at": now, "files": files}
    path.write_text(json.dumps(cache), "utf-8")
    return files


def cached_sizes(path: Path, repo: str, fetch: Callable[[str], list[dict[str, Any]]], now: float, ttl: float = 86400) -> dict[str, int]:
    return {quant: item["mb"] for quant, item in cached_files(path, repo, fetch, now, ttl).items()}


def marketplace(entry: ModelEntry, curated: set[str], files: dict[str, dict[str, Any]], hardware: dict[str, Any] | None,
                added: set[str]) -> list[dict[str, Any]]:
    """The quantizations in the model's Hub repo that the catalog file does not list, smallest first, with whether this
    machine can run them. `added` are the ones the user already added."""
    rows = []
    for quant, item in files.items():
        if quant in curated:
            continue
        need = estimate_mb(item["mb"])
        where, reason = place(need, hardware) if hardware is not None else (None, REMOTE)
        fits = True if where else (None if reason in (UNKNOWN, REMOTE) else False)
        rows.append({"quant": quant, "disk_mb": item["mb"], "estimated_mb": need, "fits": fits, "reason": "" if fits else reason,
                     "runs_on": where, "added": quant in added})
    return sorted(rows, key=lambda row: row["estimated_mb"])
