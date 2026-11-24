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
from catalog import load_catalog
from cache import AnswerCache
from disconnect import CallerGone, Measure, first_piece
from errors import CapabilityError, EngineError, UnknownModel
from options import SETTINGS
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


class QuantChoice(BaseModel):
    quant: str = Field(min_length=1, max_length=64)


class Threads(BaseModel):
    num_thread: int = Field(ge=1, le=256)


def _hub_json(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def build_router(tokens: TokenStore, config: settings.Config | None) -> ModelRouter:
    """The catalog, one adapter per engine, and the saved quantizations."""
    folder = Path(os.environ.get("SERVICE_CONFIGS_DIR", Path(__file__).resolve().parents[1] / "configs" / "models"))
    catalog = load_catalog(folder)

    def live() -> tuple[bool, bool]:
        return (str(config.get("thinking")).lower() == "on", bool(config.get("hide_reasoning")))

    adapters = {
        "ollama": OllamaAdapter(os.environ.get("SERVICE_OLLAMA_URL", "http://127.0.0.1:11434"), live=live if config else None),
        "llamacpp": LlamaCppAdapter(Path(os.environ.get("SERVICE_MODELS_DIR", "models"))),
    }
    saved = settings.load(tokens.settings_path).get("quants") or {}
    quants = {name: quant for name, quant in saved.items() if name in catalog and any(b.quant == quant for b in catalog[name].builds)}
    raw = os.environ.get("SERVICE_OLLAMA_PARALLEL", "").strip() or "2"
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError("SERVICE_OLLAMA_PARALLEL must be a whole number of 1 or more")
    return ModelRouter(catalog, adapters, quants, int(raw))


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
        nonlocal seen, threads
        while True:
            saved = await asyncio.to_thread(settings.load, tokens.settings_path)
            threads = saved.get("num_thread") or threads
            for name, quant in (saved.get("quants") or {}).items():
                if name in router.catalog and quant != router.quant_of(router.catalog[name]) and any(b.quant == quant for b in router.catalog[name].builds):
                    router.set_quant(name, quant)
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
        return [{"name": e.name, "title": e.title, "engine": e.engine, "capabilities": sorted(e.capabilities),
                 "quant": router.quant_of(e)} for e in router.catalog.values()]

    def build(reply: dict[str, Any], started: float) -> dict[str, Any]:
        calls = [{"id": call.get("id") or f"call_{index}", "name": call["function"]["name"],
                  "arguments": call["function"].get("arguments") or {}}
                 for index, call in enumerate(reply.get("tool_calls") or [])
                 if isinstance(call.get("function"), dict) and call["function"].get("name")]
        return {"content": reply["content"], "tool_calls": calls, "finish_reason": "tool_calls" if calls else reply["finish"],
                "model": reply["model"], "quant": reply["quant"],
                "usage": {"input_tokens": reply["prompt_tokens"], "output_tokens": reply["completion_tokens"]},
                "duration_ms": round((time.perf_counter() - started) * 1000)}

    def cache_key(body: ChatRequest) -> str:
        """Include the model generation so pulls invalidate old answers."""
        entry = router.entry(body.model)
        part = [router.generation, entry.name, router.quant_of(entry), [m.model_dump() for m in body.messages], body.max_tokens,
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
            router.entry(body.model)
        except UnknownModel:
            raise HTTPException(404, "Unknown model") from None
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
                events = router.chat_stream(body.model, messages, temperature=body.temperature, max_tokens=body.max_tokens, **extra)
                try:
                    first = await first_piece(request, events, None)  # waited for here, so a refusal is still an HTTP error
                except CallerGone:
                    await events.aclose()
                    return Response(status_code=499)  # nobody is listening; the generation is already stopped
                except BaseException:
                    await events.aclose()
                    raise
                return ndjson(live_lines(events, first, started, key, size), "miss")
            reply = await router.chat(body.model, messages, temperature=body.temperature, max_tokens=body.max_tokens, **extra)
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
        """Every model with its builds under it: size, fit on this machine, downloaded. Read when asked, never cached."""
        probed: dict[bool, Any] = {}
        sizes_file = Path(tokens.settings_path).parent / "hub-sizes.json"
        rows = []
        for entry in router.catalog.values():
            adapter = router.adapters[entry.engine]
            if adapter.same_machine and True not in probed:
                probed[True] = await asyncio.to_thread(hardware.probe)
            hw = probed.get(True) if adapter.same_machine else None
            try:
                have = await asyncio.to_thread(adapter.downloaded)
            except EngineError:
                have = set()
            live = await asyncio.to_thread(builds.cached_sizes, sizes_file, entry.repo, fetch_sizes, time.time())
            current = router.quant_of(entry)
            rows.append({"name": entry.name, "title": entry.title, "engine": entry.engine, "capabilities": sorted(entry.capabilities),
                         "current_quant": current,
                         "builds": builds.listing(entry, hw, current, have, live)})
        return probed.get(True), rows

    @app.get("/admin/models", dependencies=[Depends(admin)])
    async def list_models() -> dict[str, Any]:
        hw, rows = await model_rows()
        return {"hardware": hw, "summary": hardware.describe(hw), "models": rows, "estimate": "Estimated from the file size; not measured."}

    def pick(name: str, quant: str):
        try:
            entry = router.entry(name)
        except UnknownModel:
            raise HTTPException(404, "No such model.") from None
        try:
            entry.build(quant)
        except KeyError:
            raise HTTPException(422, "No such build for this model.") from None
        return entry

    @app.put("/admin/models/{name}/quant", dependencies=[Depends(admin)])
    async def choose_quant(name: str, body: QuantChoice) -> dict[str, Any]:
        """Use this build of the model from now on; every replica applies it within a second."""
        entry = pick(name, body.quant)
        row = next(r for r in next(m for m in (await model_rows())[1] if m["name"] == name)["builds"] if r["id"] == body.quant)
        if row["fits"] is False:
            raise HTTPException(409, row["reason"])
        router.set_quant(name, body.quant)
        saved = settings.load(tokens.settings_path).get("quants") or {}
        settings.save(tokens.settings_path, quants={**saved, name: body.quant})
        return {"saved": body.quant, "downloaded": row["downloaded"]}

    @app.post("/admin/models/{name}/download", dependencies=[Depends(admin)])
    async def download_build(name: str, body: QuantChoice) -> dict[str, Any]:
        """The only place weights are fetched. Every replica then drops its loaded copy."""
        entry = pick(name, body.quant)
        try:
            await asyncio.to_thread(router.adapters[entry.engine].download, entry, body.quant)
        except EngineError as error:
            raise HTTPException(502, f"Download failed: {error}") from None
        await unload_all()
        return {"downloaded": entry.source(body.quant)}

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
