"""HTTP API for the chat models in the catalog, with shared admin, tokens, cache and streaming."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import urllib.request
import uuid
import logging
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

import builds
import hardware
from adapters.llamacpp import LlamaCppAdapter
from adapters.ollama import OllamaAdapter
from catalog import load_catalog, with_extras
from cache import AnswerCache
from disconnect import CallerGone, Measure, first_piece
from dataclasses import replace
from errors import CapabilityError, EngineError, UnknownModel
from options import SETTINGS
from progress import progress
from router import ModelRouter
import settings
from tokens import LatencyHistogram, TokenStore

_log = logging.getLogger(__name__)

class ToolCall(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """`tool_calls` belongs to an assistant turn; a `tool` message carries one call's result (`tool_name` names it)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field("", max_length=100_000)
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = Field(None, max_length=64)


class Tool(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    description: str = Field("", max_length=2000)
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # schema, project and thinking are refused, not silently ignored

    model: str = Field(min_length=1, max_length=64)  # a catalog name; every request names its model
    engine: str | None = Field(None, max_length=32)  # one the model's current build lists; none = the user's saved choice
    messages: list[Message] = Field(min_length=1, max_length=64)
    tools: list[Tool] | None = Field(None, max_length=32)
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(512, ge=1, le=4096)
    stream: bool = False  # true: the answer arrives as JSON lines while it is made


def _ollama_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [{"function": {"name": call.name, "arguments": call.arguments}} for call in message.tool_calls]
    if message.tool_name:
        out["tool_name"] = message.tool_name
    return out


def _ollama_tool(tool: Tool) -> dict[str, Any]:
    return {"type": "function", "function": tool.model_dump()}


class SettingsUpdate(BaseModel):
    values: dict[str, Any]  # None resets an option to its environment variable or default


class NewToken(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    projects: None = None  # null-only compatibility with the shared token editor
    rate: int | None = Field(None, ge=1, le=100_000)  # calls per minute
    expires_days: int | None = Field(None, ge=1, le=3650)


class TokenScope(BaseModel):
    projects: None = None  # null-only compatibility with the shared token editor
    rate: int | None = Field(None, ge=1, le=100_000)


class Variant(BaseModel):
    quant: str = Field(min_length=1, max_length=64)


class Choice(BaseModel):
    quant: str = Field(min_length=1, max_length=64)
    engine: str = Field(min_length=1, max_length=32)


class Threads(BaseModel):
    num_thread: int = Field(ge=1, le=256)


def _hub_json(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def _offline(_url: str) -> list[dict[str, Any]]:
    raise OSError("offline")  # used where only the cache may answer


def read_extras(tokens: TokenStore) -> dict[str, list[dict[str, Any]]]:
    """The quantizations the user added from the marketplace: {model: [{quant, file_mb}]}, shared by every replica."""
    try:
        data = json.loads((Path(tokens.settings_path).parent / "marketplace.json").read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def build_router(tokens: TokenStore, config: settings.Config | None) -> ModelRouter:
    """The catalog, one adapter per engine, and the saved build and engine choices."""
    folder = Path(os.environ.get("SERVICE_CONFIGS_DIR", Path(__file__).resolve().parents[1] / "configs" / "models"))
    extras = read_extras(tokens)
    catalog = {name: with_extras(entry, extras.get(name, [])) for name, entry in load_catalog(folder).items()}

    def live() -> tuple[bool, bool]:
        return (str(config.get("thinking")).lower() == "on", bool(config.get("hide_reasoning")))

    def option(key: str) -> Any:
        """The saved admin value, else the environment variable, else the default (without admin settings: env or default)."""
        if config:
            return config.get(key)
        spec = next(spec for spec in SETTINGS if spec.key == key)
        return type(spec.default)(os.environ.get(spec.env) or spec.default)

    adapters = {"ollama": OllamaAdapter(option("ollama_url"), live=live if config else None),
                "llamacpp": LlamaCppAdapter(Path(option("models_dir")))}
    saved = settings.load(tokens.settings_path).get("quants") or {}
    quants = {name: quant for name, quant in saved.items() if name in catalog and any(b.quant == quant for b in catalog[name].builds)}
    engines = {name: engine for name, engine in (settings.load(tokens.settings_path).get("engines") or {}).items()
               if name in catalog and engine in catalog[name].engines(quants.get(name, catalog[name].default_quant))}
    router = ModelRouter(catalog, adapters, quants, option("ollama_parallel"), engines)
    router.stopped = set(settings.load(tokens.settings_path).get("stopped") or []) & set(catalog)
    return router


def create_app(router: ModelRouter | None = None, admin_password: str | None = None,
               tokens: TokenStore | None = None, max_inflight: int | None = None,
               cache_size: int | None = None, config: settings.Config | None = None,
               fetch_sizes: Any = _hub_json) -> FastAPI:
    """`admin_password` (SERVICE_ADMIN_PASSWORD) unlocks /admin/, where each project gets a token of its own; when it
    is set, every call except /health must send one of those tokens in the X-Internal-Token header. Unset = no
    auth at all (local use only). `tokens` holds the project tokens (SERVICE_TOKENS_FILE). `cache_size` > 0 keeps that
    many answers per replica (SERVICE_CACHE_SIZE). `config` (the admin page's settings) replaces
    `max_inflight` and `cache_size` and keeps them live; without it they stay fixed. `fetch_sizes(url)` reads the Hub file tree."""
    tokens = tokens or TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    router = router or build_router(tokens, config)
    inflight = 0
    cache = AnswerCache()
    sizes_file = Path(tokens.settings_path).parent / "hub-sizes.json"
    extras_file = Path(tokens.settings_path).parent / "marketplace.json"
    extras = read_extras(tokens)
    extras_text = extras_file.read_text("utf-8") if extras_file.exists() else ""

    def curated(entry):
        """The model as its catalog file lists it, without the marketplace additions."""
        added = {x["quant"] for x in extras.get(entry.name, [])}
        return replace(entry, builds=tuple(b for b in entry.builds if b.quant not in added))

    base_catalog = {name: curated(entry) for name, entry in router.catalog.items()}

    def apply_extras() -> None:
        router.catalog.update({name: with_extras(entry, extras.get(name, [])) for name, entry in base_catalog.items()})

    def write_extras() -> None:
        nonlocal extras_text
        extras_text = json.dumps(extras)
        extras_file.write_text(extras_text, "utf-8")

    def fit_reason(entry, quant: str, engine: str) -> str | None:
        """Why this build cannot run on this machine, or None (it fits, or the engine is elsewhere or memory is unreadable).
        Sizes come from the cache only: no network call on the request path."""
        if not router.adapters[engine].same_machine:
            return None
        files = builds.cached_files(sizes_file, entry.repo, _offline, time.time())
        mb = files.get(quant, {}).get("mb") or entry.build(quant).file_mb
        where, reason = builds.place(builds.estimate_mb(mb), hardware.probe())
        return None if where or reason == builds.UNKNOWN else reason

    router.guard = router.guard or fit_reason  # a build that does not fit is never loaded, not even by a request
    marker = Path(tokens.settings_path).parent / ".cnv-weights-version"
    threads: int | None = None  # saved and reported only; Ollama chooses its own threads

    def cache_limit() -> int:
        """Answers this replica keeps; 0 = off."""
        return config.get("cache_size") if config else (
            cache_size if cache_size is not None else int(os.environ.get("SERVICE_CACHE_SIZE") or 0))

    def inflight_limit() -> int:
        """Calls this replica lets wait for a model at once; more get 429. 0 = 4 per parallel slot."""
        value = config.get("max_inflight") if config else (
            max_inflight if max_inflight is not None else int(os.environ.get("SERVICE_MAX_INFLIGHT") or 0))
        return value or 4 * router.parallel

    def bump_marker() -> None:
        """Other replicas drop their loaded models when this file's content changes."""
        marker.write_text(uuid.uuid4().hex, "utf-8")

    def read_marker() -> str | None:
        try:
            return marker.read_text("utf-8")
        except OSError:
            return None

    seen = read_marker()

    async def unload_all() -> None:
        await asyncio.to_thread(bump_marker)
        router.forget()

    async def sync_forever() -> None:
        """Applies saved admin values and drops models another replica replaced with new weights."""
        nonlocal seen, threads, extras_text
        while True:
            saved = await asyncio.to_thread(settings.load, tokens.settings_path)
            threads = saved.get("num_thread") or threads
            text = extras_file.read_text("utf-8") if extras_file.exists() else ""
            if text != extras_text:
                extras_text = text
                extras.clear()
                extras.update(read_extras(tokens))
                apply_extras()
            stopped = set(saved.get("stopped") or []) & set(router.catalog)
            for name in stopped - router.stopped:
                router.forget(name)
            router.stopped = stopped
            ollama, llama = router.adapters["ollama"], router.adapters["llamacpp"]
            if (url := config.get("ollama_url").strip().rstrip("/")) != ollama.url:
                ollama.url = url
                router.forget()
            if (folder := Path(config.get("models_dir"))) != llama.models_dir:
                llama.models_dir = folder
                router.forget()
            if config.get("ollama_parallel") != router.parallel:
                router.set_parallel(config.get("ollama_parallel"))
            for name in router.catalog:
                entry = router.catalog[name]
                quant = (saved.get("quants") or {}).get(name, entry.default_quant)
                engine = (saved.get("engines") or {}).get(name)
                if quant in {b.quant for b in entry.builds} and engine in entry.engines(quant) \
                        and (quant, engine) != (router.quant_of(entry), router.engines.get(name)):
                    router.set_choice(name, quant, engine)
            current = await asyncio.to_thread(read_marker)
            if current != seen:
                seen = current
                router.forget()
            await asyncio.sleep(1)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        background: list[asyncio.Task] = []
        try:
            if config:
                background.append(asyncio.create_task(sync_forever()))
            yield
        finally:
            for task in background:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await router.close()

    def authorize(request: Request, x_internal_token: str = Header("")) -> None:
        """Authenticate and rate-limit public tokens. Legacy scope lists have no effect."""
        if not admin_password:
            return None
        row = tokens.lookup(x_internal_token)
        if row is None:
            raise HTTPException(401, "Invalid token")
        wait = tokens.throttle(row)
        if wait:
            raise HTTPException(429, "Too many requests for this token", headers={"Retry-After": str(wait)})
        request.state.token_id = row["id"]

    def admin(x_internal_token: str = Header("")) -> None:
        if not admin_password or not hmac.compare_digest(x_internal_token, admin_password):
            raise HTTPException(401, "Admin password required (set SERVICE_ADMIN_PASSWORD)")

    app = FastAPI(title="CNV chat model service", lifespan=lifespan)

    async def slot():
        nonlocal inflight
        limit = inflight_limit()
        if limit and inflight >= limit:
            raise HTTPException(429, "The chat models are busy: too many requests waiting", headers={"Retry-After": "5"})
        inflight += 1
        try:
            yield
        finally:
            inflight -= 1

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "ready": router.ready, "models": list(router.catalog), "num_thread": threads}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> list[dict[str, Any]]:
        return [{"name": e.name, "title": e.title, "capabilities": sorted(e.capabilities), "quant": router.quant_of(e),
                 "engine": router.engine_of(e), "state": router.state_of(e.name), "builds": [{"quant": b.quant, "engines": list(b.engines)} for b in e.builds]}
                for e in router.catalog.values()]

    def build(reply: dict[str, Any], started: float) -> dict[str, Any]:
        calls = [{"id": call.get("id") or f"call_{index}", "name": call["function"]["name"],
                  "arguments": call["function"].get("arguments") or {}}
                 for index, call in enumerate(reply.get("tool_calls") or [])
                 if isinstance(call.get("function"), dict) and call["function"].get("name")]
        return {"content": reply["content"], "tool_calls": calls, "finish_reason": "tool_calls" if calls else reply["finish"],
                "model": reply["model"], "quant": reply["quant"], "engine": reply["engine"],
                "usage": {"input_tokens": reply["prompt_tokens"], "output_tokens": reply["completion_tokens"]},
                "duration_ms": round((time.perf_counter() - started) * 1000)}

    def cache_key(body: ChatRequest) -> str:
        """Include the model generation so pulls invalidate old answers."""
        entry = router.entry(body.model)
        part = [router.generation, entry.name, router.quant_of(entry), router.engine_of(entry, body.engine), [m.model_dump() for m in body.messages], body.max_tokens,
                [t.model_dump() for t in body.tools or []],
                config.get("thinking") if config else os.environ.get("SERVICE_THINKING", "off"),
                config.get("hide_reasoning") if config else os.environ.get("SERVICE_HIDE_REASONING", "true")]
        return hashlib.sha256(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()

    def line(event: dict[str, Any]) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    def ndjson(lines: Any, cached: str) -> StreamingResponse:
        return StreamingResponse(lines, media_type="application/x-ndjson", headers={"X-Cache": cached})

    async def replay(result: dict[str, Any]):
        """A kept answer in the shape of a stream: the whole text in one piece, then `done`."""
        if result["content"]:
            yield line({"type": "content", "text": result["content"]})
        yield line({"type": "done", **result})

    async def live_lines(events: Any, first: tuple[str, Any] | None, started: float, key: str | None, size: int):
        """One JSON line per piece of text, then `done` with the same body a call without `stream` returns. A failure
        after the first piece can only be sent as an `error` line."""
        parts: list[str] = []
        stats: dict[str, Any] = {}
        calls: list[dict[str, Any]] = []
        try:
            async def all_events():
                if first is not None:
                    yield first
                async for event in events:
                    yield event

            async for kind, value in all_events():
                if kind == "content":
                    parts.append(value)
                    yield line({"type": "content", "text": value})
                elif kind == "tool_calls":
                    calls.extend(value)
                else:
                    stats = value
        except (UnknownModel, CapabilityError) as error:
            yield line({"type": "error", "status": 404 if isinstance(error, UnknownModel) else 422, "message": str(error)})
            return
        except EngineError as error:
            yield line({"type": "error", "status": 422 if error.status == 400 else error.status or 500, "message": str(error)})
            return
        except Exception:
            _log.exception("Chat failed")
            yield line({"type": "error", "status": 500, "message": "The model could not answer"})
            return
        finally:
            await events.aclose()
        result = build({**stats, "content": "".join(parts).strip(), "tool_calls": calls}, started)
        if key:
            cache.put(key, result, size)
        yield line({"type": "done", **result})

    @app.post("/v1/chat")
    async def chat(body: ChatRequest, request: Request, response: Response, _auth: None = Depends(authorize),
                   _slot: None = Depends(slot)) -> Any:
        started = time.perf_counter()
        try:
            router.engine_of(router.entry(body.model), body.engine)
        except UnknownModel:
            raise HTTPException(404, "Unknown model") from None
        except CapabilityError as error:
            raise HTTPException(422, str(error)) from None
        size = cache_limit() if body.temperature == 0 else 0  # a higher temperature asks for a different answer each time
        key = cache_key(body) if size else None
        kept = cache.get(key, size) if key else None
        if kept is not None:
            kept = {**kept, "duration_ms": round((time.perf_counter() - started) * 1000)}
            if body.stream:
                return ndjson(replay(kept), "hit")
            response.headers["X-Cache"] = "hit"
            return kept
        messages = [_ollama_message(m) for m in body.messages]
        extra = {"tools": [_ollama_tool(t) for t in body.tools]} if body.tools else {}  # only sent when used
        try:
            if body.stream:
                events = router.chat_stream(body.model, messages, engine=body.engine, temperature=body.temperature, max_tokens=body.max_tokens, **extra)
                try:
                    first = await first_piece(request, events, None)  # waited for here, so a refusal is still an HTTP error
                except CallerGone:
                    await events.aclose()
                    return Response(status_code=499)  # nobody is listening; the generation is already stopped
                except BaseException:
                    await events.aclose()
                    raise
                return ndjson(live_lines(events, first, started, key, size), "miss")
            reply = await router.chat(body.model, messages, engine=body.engine, temperature=body.temperature, max_tokens=body.max_tokens, **extra)
        except CapabilityError as error:
            raise HTTPException(422, str(error)) from None
        except EngineError as error:
            raise HTTPException(422 if error.status == 400 else error.status or 500, str(error)) from None
        except Exception:
            _log.exception("Chat failed")
            raise HTTPException(500, "The model could not answer") from None
        result = build(reply, started)
        if key:
            cache.put(key, result, size)
        response.headers["X-Cache"] = "miss"
        return result

    @app.put("/admin/threads", dependencies=[Depends(admin)])
    def set_threads(body: Threads) -> dict[str, Any]:
        """Save and report the value for compatibility; Ollama chooses its own threads."""
        nonlocal threads
        threads = body.num_thread
        settings.save(tokens.settings_path, num_thread=body.num_thread)
        return {"num_thread": threads}

    async def model_rows() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Every model with its builds under it, and under each build the engines that can run it: size, fit, downloaded.
        Read when asked, never cached."""
        machine = await asyncio.to_thread(hardware.probe) if any(a.same_machine for a in router.adapters.values()) else None
        on_disk: dict[str, set[str]] = {}
        for engine, adapter in router.adapters.items():
            try:
                on_disk[engine] = await asyncio.to_thread(adapter.downloaded)
            except EngineError:
                on_disk[engine] = set()
        hw = {engine: machine if adapter.same_machine else None for engine, adapter in router.adapters.items()}
        rows = []
        for entry in router.catalog.values():
            live = await asyncio.to_thread(builds.cached_sizes, sizes_file, entry.repo, fetch_sizes, time.time())
            current = (router.quant_of(entry), router.engine_of(entry))
            rows.append({"name": entry.name, "title": entry.title, "capabilities": sorted(entry.capabilities),
                         "current_quant": current[0], "current_engine": current[1], "state": router.state_of(entry.name),
                         "builds": builds.listing(entry, hw, current, on_disk, live)})
        return machine, rows

    @app.get("/admin/models", dependencies=[Depends(admin)])
    async def list_models() -> dict[str, Any]:
        hw, rows = await model_rows()
        return {"hardware": hw, "summary": hardware.describe(hw), "models": rows, "estimate": "Estimated from the file size; not measured."}

    def pick(name: str, body: Choice):
        try:
            entry = router.entry(name)
        except UnknownModel:
            raise HTTPException(404, "No such model.") from None
        try:
            engines = entry.engines(body.quant)
        except KeyError:
            raise HTTPException(422, "No such build for this model.") from None
        if body.engine not in engines:
            raise HTTPException(422, f"This build runs on: {', '.join(engines)}.")
        return entry

    @app.put("/admin/models/{name}/run", dependencies=[Depends(admin)])
    async def choose_run(name: str, body: Choice) -> dict[str, Any]:
        """Run this model as this build on this engine from now on; every replica applies it within a second."""
        pick(name, body)
        model = next(m for m in (await model_rows())[1] if m["name"] == name)
        option = next(o for r in model["builds"] if r["id"] == body.quant for o in r["engines"] if o["engine"] == body.engine)
        if option["fits"] is False:
            raise HTTPException(409, option["reason"])
        router.set_choice(name, body.quant, body.engine)
        saved = settings.load(tokens.settings_path)
        settings.save(tokens.settings_path, quants={**(saved.get("quants") or {}), name: body.quant},
                      engines={**(saved.get("engines") or {}), name: body.engine})
        return {"saved": {"quant": body.quant, "engine": body.engine}, "downloaded": option["downloaded"]}

    @app.post("/admin/models/{name}/download", dependencies=[Depends(admin)])
    async def download_build(name: str, body: Choice) -> dict[str, Any]:
        """The only place weights are fetched. Every replica then drops its loaded copy."""
        entry = pick(name, body)
        if progress.snapshot()["active"]:
            raise HTTPException(409, "Another download is running; wait for it to finish.")
        if (body.quant, body.engine) != (router.quant_of(entry), router.engine_of(entry)) \
                and (reason := await asyncio.to_thread(fit_reason, entry, body.quant, body.engine)):
            raise HTTPException(409, reason)  # only builds this machine can run are downloaded
        try:
            await asyncio.to_thread(router.adapters[body.engine].download, entry, body.quant)
        except EngineError as error:
            raise HTTPException(502, f"Download failed: {error}") from None
        await unload_all()
        return {"downloaded": entry.source(body.quant, body.engine)}

    @app.get("/admin/progress", dependencies=[Depends(admin)])
    def download_progress() -> dict[str, Any]:
        """The running download: {active, label, done, total} in bytes (total is null while unknown)."""
        return progress.snapshot()

    @app.post("/admin/models/{name}/delete", dependencies=[Depends(admin)])
    async def delete_build(name: str, body: Choice) -> dict[str, Any]:
        """Delete a downloaded build from its engine's storage. The build a running model uses is kept until the model is stopped."""
        entry = pick(name, body)
        if progress.snapshot()["active"]:
            raise HTTPException(409, "A download is running; wait for it to finish.")
        if (body.quant, body.engine) == (router.quant_of(entry), router.engine_of(entry)) and name not in router.stopped:
            raise HTTPException(409, "This is the build the model runs: stop the model or run another build first.")
        try:
            await asyncio.to_thread(router.adapters[body.engine].delete, entry, body.quant)
        except EngineError as error:
            raise HTTPException(502, f"Delete failed: {error}") from None
        await unload_all()
        return {"deleted": entry.source(body.quant, body.engine)}

    def persist_stopped() -> None:
        settings.save(tokens.settings_path, stopped=sorted(router.stopped))

    @app.post("/admin/models/{name}/{action}", dependencies=[Depends(admin)])
    async def lifecycle(name: str, action: Literal["start", "stop", "restart"]) -> dict[str, Any]:
        """Stop frees the model and refuses its requests; start loads it and allows them again; restart is both."""
        try:
            entry = router.entry(name)
        except UnknownModel:
            raise HTTPException(404, "No such model.") from None
        if action in ("stop", "restart"):
            await router.stop(name)
            persist_stopped()
        if action in ("start", "restart"):
            if reason := await asyncio.to_thread(fit_reason, entry, router.quant_of(entry), router.engine_of(entry)):
                raise HTTPException(409, reason)  # stays stopped
            try:
                await router.start(name)
            except EngineError as error:
                persist_stopped()
                raise HTTPException(error.status or 502, f"Started, but it could not be loaded now: {error}") from None
            persist_stopped()
        return {"state": router.state_of(name)}

    async def marketplace_files(entry) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(builds.cached_files, sizes_file, entry.repo, fetch_sizes, time.time())

    def ollama_machine() -> dict[str, Any] | None:
        return hardware.probe() if router.adapters["ollama"].same_machine else None

    @app.get("/admin/marketplace", dependencies=[Depends(admin)])
    async def list_marketplace() -> dict[str, Any]:
        """Quantizations in each model's Hub repo that the catalog does not list, with whether this machine can run them."""
        machine = await asyncio.to_thread(ollama_machine)
        models = []
        for name, base in base_catalog.items():
            files = await marketplace_files(base)
            runs_on_ollama = "ollama" in base.engines(base.default_quant)
            rows = builds.marketplace(base, {b.quant for b in base.builds}, files, machine, {x["quant"] for x in extras.get(name, [])}) if runs_on_ollama else []
            note = "" if runs_on_ollama else "Runs on its own sidecar: add builds in its catalog file."
            models.append({"name": name, "title": base.title, "variants": rows,
                           "note": note or ("" if files else "Could not read the Hub listing (offline?).")})
        return {"summary": hardware.describe(machine), "models": models, "estimate": "Estimated from the file size; not measured."}

    @app.post("/admin/marketplace/{name}/add", dependencies=[Depends(admin)])
    async def marketplace_add(name: str, body: Variant) -> dict[str, Any]:
        """Add a Hub quantization to the model's builds. Only builds this machine can run are added."""
        try:
            base = base_catalog[router.entry(name).name]
        except UnknownModel:
            raise HTTPException(404, "No such model.") from None
        if "ollama" not in base.engines(base.default_quant):
            raise HTTPException(422, "This model runs on its own sidecar: add builds in its catalog file.")
        item = (await marketplace_files(base)).get(body.quant)
        if item is None or body.quant in {b.quant for b in base.builds}:
            raise HTTPException(404, "The Hub repo does not offer this variant, or the catalog lists it already.")
        row = builds.marketplace(base, set(), {body.quant: item}, await asyncio.to_thread(ollama_machine), set())[0]
        if row["fits"] is False:
            raise HTTPException(409, row["reason"])
        if body.quant not in {x["quant"] for x in extras.get(name, [])}:
            extras.setdefault(name, []).append({"quant": body.quant, "file_mb": item["mb"]})
            write_extras()
            apply_extras()
        return {"added": body.quant}

    @app.delete("/admin/marketplace/{name}/{quant}", dependencies=[Depends(admin)])
    async def marketplace_remove(name: str, quant: str) -> dict[str, Any]:
        """Take an added variant out of the model's builds. Its downloaded files stay on disk."""
        if quant not in {x["quant"] for x in extras.get(name, [])}:
            raise HTTPException(404, "No such added variant.")
        if router.quant_of(router.entry(name)) == quant:
            raise HTTPException(409, "It is the build this model runs: run another build first.")
        extras[name] = [x for x in extras[name] if x["quant"] != quant]
        write_extras()
        apply_extras()
        return {"removed": quant}

    @app.get("/admin/workers", dependencies=[Depends(admin)])
    def workers() -> dict[str, Any]:
        """Read-only: the replica count is a Docker Compose setting the service cannot change."""
        def number(name: str) -> int | None:
            value = os.environ.get(name, "").strip()
            return int(value) if value.isdigit() else None

        return {"replicas": number("SERVICE_REPLICAS"), "ollama_parallel": router.parallel}

    @app.get("/admin/tokens", dependencies=[Depends(admin)])
    def list_tokens() -> dict[str, Any]:
        return {"tokens": [{**row, "projects": None} for row in tokens.list()]}

    @app.post("/admin/tokens", dependencies=[Depends(admin)])
    def create_token(body: NewToken) -> dict[str, Any]:
        return tokens.create(body.name.strip(), None, body.rate, body.expires_days)

    @app.get("/admin/settings", dependencies=[Depends(admin)])
    def get_settings() -> dict[str, Any]:
        return {"settings": config.describe() if config else []}

    @app.put("/admin/settings", dependencies=[Depends(admin)])
    def put_settings(body: SettingsUpdate) -> dict[str, Any]:
        """Saves options; null resets one. Every replica reads the shared file, so a change applies everywhere within
        a second, except options marked `restart`."""
        if config is None:
            raise HTTPException(409, "Settings are fixed in this setup")
        if (url := body.values.get("ollama_url")) and not str(url).startswith(("http://", "https://")):
            raise HTTPException(422, "The Ollama server address must start with http:// or https://")
        try:
            config.set(body.values)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {"settings": config.describe()}

    @app.put("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def update_token(token_id: str, body: TokenScope) -> dict[str, bool]:
        if not tokens.update(token_id, None, body.rate):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    latency = LatencyHistogram()

    app.add_middleware(Measure, observe=latency.observe,
                       record=lambda token_id, ms, failed: run_in_threadpool(tokens.record, token_id, ms, failed))

    @app.get("/metrics", dependencies=[Depends(admin)])
    def metrics() -> PlainTextResponse:
        """Prometheus text per token, from the shared token file (up to a minute behind). Send the admin password."""
        return PlainTextResponse(tokens.metrics("cnv") + latency.render("cnv"))

    @app.delete("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def revoke_token(token_id: str) -> dict[str, bool]:
        if not tokens.revoke(token_id):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    web = Path(os.environ.get("SERVICE_WEB_DIR", Path(__file__).resolve().parents[1] / "web"))
    if web.is_dir():
        app.mount("/admin", StaticFiles(directory=web, html=True))

    return app


def main() -> None:
    import sys

    import uvicorn

    if "--dotenv" in sys.argv[1:]:  # run.bat asks for it; a plain `python src/app.py` and Docker never read .env files
        loaded = settings.load_env_files(Path(__file__).resolve().parents[1])
        for name, names in loaded.items():
            print(f"{name}: loaded {', '.join(names) or 'no variables'} (names only; a real environment variable wins)", flush=True)
        if not loaded:
            print(".env.local / .env: no file in the service folder", flush=True)
    store = TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    config = settings.Config(store.settings_path, SETTINGS)
    router = build_router(store, config)
    app = create_app(router=router, admin_password=os.environ.get("SERVICE_ADMIN_PASSWORD") or None,
                     tokens=store, config=config)
    host, port = os.environ.get("SERVICE_HOST", "127.0.0.1"), int(os.environ.get("SERVICE_PORT", "29204"))

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)  # after uvicorn's own "running on" line
            shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
            print(f"INFO:     Admin page: http://{shown}:{port}/admin/", flush=True)

    Server(uvicorn.Config(app, host=host, port=port)).run()


if __name__ == "__main__":
    main()
