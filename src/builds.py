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


def listing(entry: ModelEntry, hardware: dict[str, Any] | None, current: str,
            downloaded: set[str], live: dict[str, int]) -> list[dict[str, Any]]:
    """Every build of `entry`, smallest first. `hardware` is None when the engine is on another machine."""
    rows = []
    for build in entry.builds:
        mb = live.get(build.quant, build.file_mb)
        need, name = estimate_mb(mb), entry.source(build.quant)
        in_use, done = build.quant == current, name in downloaded
        where, reason = place(need, hardware) if hardware is not None else (None, REMOTE)
        fits = True if in_use or where else (None if reason in (UNKNOWN, REMOTE) else False)
        runs = {"gpu": " · runs on the GPU", "cpu": " · runs on the CPU (slower)"}.get(where, "")
        rows.append({"id": build.quant, "title": build.quant, "quant": build.quant, "model": name, "note": build.note,
                     "detail": f"about {need} MB · {mb} MB download{' · downloaded' if done else ''}{runs} · {build.note}",
                     "disk_mb": mb, "estimated_mb": need, "runs_on": where, "fits": fits,
                     "reason": "" if in_use or fits else reason, "downloaded": done, "current": in_use})
    return sorted(rows, key=lambda row: row["estimated_mb"])


def _quant_key(text: str) -> str:
    return text if text[0] == "i" else text.upper()


def hub_sizes(repo: str, fetch: Callable[[str], list[dict[str, Any]]]) -> dict[str, int]:
    """Quant -> size in MB for the GGUF files of a Hub repo. `fetch(url)` returns the decoded file-tree JSON."""
    sizes = {}
    for item in fetch(f"https://huggingface.co/api/models/{repo}/tree/main"):
        match = _QUANT.search(item.get("path", ""))
        if match and item.get("size"):
            sizes[_quant_key(match.group(1))] = round(item["size"] / 1_048_576)
    return sizes


def cached_sizes(path: Path, repo: str, fetch: Callable[[str], list[dict[str, Any]]], now: float, ttl: float = 86400) -> dict[str, int]:
    """Live sizes with a JSON cache in the data volume. A fresh cache skips the Hub; a Hub failure keeps the old cache."""
    try:
        cache = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(repo)
    if hit and now - hit["at"] < ttl:
        return hit["sizes"]
    try:
        sizes = hub_sizes(repo, fetch)
    except (OSError, ValueError):
        return hit["sizes"] if hit else {}
    cache[repo] = {"at": now, "sizes": sizes}
    path.write_text(json.dumps(cache), "utf-8")
    return sizes
