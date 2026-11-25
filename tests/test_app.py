import pytest
from fastapi.testclient import TestClient

from app import create_app
from conftest import make_router
from errors import EngineError
from tokens import TokenStore

BODY = {"model": "lfm3", "messages": [{"role": "user", "content": "hi"}]}
TOOL = {"name": "f", "parameters": {"type": "object", "properties": {}}}
ADMIN = {"X-Internal-Token": "pw"}


def no_hub(_url):
    raise OSError("offline")


def client(router=None, admin_password=None, **options):
    return TestClient(create_app(router or make_router(), admin_password, fetch_sizes=no_hub, **options))


def test_chat_reports_model_quant_and_usage():
    body = client().post("/v1/chat", json={**BODY, "temperature": 0.3, "max_tokens": 50}).json()
    assert body["content"] == "hello" and body["finish_reason"] == "stop"
    assert body["model"] == "lfm3" and body["quant"] == "Q4_K_M"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 5} and "duration_ms" in body


def test_chat_picks_the_named_model_and_engine():
    router = make_router()
    body = client(router).post("/v1/chat", json={**BODY, "model": "bitnet2b"}).json()
    assert body["model"] == "bitnet2b" and router.adapters["llamacpp"].loads == [("bitnet2b", "i2_s")]


def test_chat_without_model_is_422():
    assert client().post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 422


def test_chat_unknown_model_is_404():
    assert client().post("/v1/chat", json={**BODY, "model": "zzz"}).status_code == 404


def test_tools_on_a_model_without_tools_is_422():
    assert client().post("/v1/chat", json={**BODY, "model": "bitnet2b", "tools": [TOOL]}).status_code == 422


@pytest.mark.parametrize("extra", [{"schema": {"type": "object"}}, {"thinking": False}, {"project": "x"}])
def test_features_the_service_lacks_are_refused_not_ignored(extra):
    assert client().post("/v1/chat", json={**BODY, **extra}).status_code == 422


def test_engine_error_status_is_passed_on():
    assert client(make_router(fail=EngineError("down", 503))).post("/v1/chat", json=BODY).status_code == 503


def test_stream_ends_with_done():
    lines = client().post("/v1/chat", json={**BODY, "stream": True}).text.strip().splitlines()
    assert '"type": "content"' in lines[0] and '"type": "done"' in lines[-1] and '"model": "lfm3"' in lines[-1]


def test_stream_unknown_model_is_404():
    assert client().post("/v1/chat", json={**BODY, "model": "zzz", "stream": True}).status_code == 404


def test_models_lists_capabilities_and_engines_per_build():
    rows = client().get("/v1/models").json()
    lfm3 = next(r for r in rows if r["name"] == "lfm3")
    assert lfm3["engine"] == "ollama" and {"quant": "Q8_0", "engines": ["ollama"]} in lfm3["builds"]
    assert "tools" in lfm3["capabilities"] and {r["name"] for r in rows} >= {"bitnet2b", "minicpm52b", "sparkx26"}


def test_admin_models_lists_builds_under_each_model():
    c = client(admin_password="pw")
    assert c.get("/admin/models").status_code == 401
    rows = c.get("/admin/models", headers=ADMIN).json()["models"]
    lfm3 = next(r for r in rows if r["name"] == "lfm3")
    assert [b["id"] for b in lfm3["builds"]][0] == "Q4_K_M"
    assert lfm3["current_quant"] == "Q4_K_M" and lfm3["current_engine"] == "ollama"
    assert [e["engine"] for e in lfm3["builds"][0]["engines"]] == ["ollama"] and "downloaded" in lfm3["builds"][0]["engines"][0]


def test_run_saves_the_build_and_engine_the_user_chose(tmp_path):
    router = make_router()
    store = TokenStore(str(tmp_path / "tokens.json"))
    c = TestClient(create_app(router, "pw", tokens=store, fetch_sizes=no_hub))

    def run(name, quant, engine):
        return c.put(f"/admin/models/{name}/run", json={"quant": quant, "engine": engine}, headers=ADMIN)

    assert run("lfm3", "Q1", "ollama").status_code == 422
    assert run("lfm3", "Q8_0", "llamacpp").status_code == 422  # the build does not list that engine
    assert run("zzz", "Q4_K_M", "ollama").status_code == 404
    assert run("lfm3", "Q8_0", "ollama").status_code == 200
    token = {"X-Internal-Token": store.create("t")["token"]}
    body = c.post("/v1/chat", json=BODY, headers=token).json()
    assert (body["quant"], body["engine"]) == ("Q8_0", "ollama")


def test_request_engine_must_be_listed_by_the_build():
    c = client()
    assert c.post("/v1/chat", json={**BODY, "engine": "llamacpp"}).status_code == 422
    assert c.post("/v1/chat", json={**BODY, "engine": "ollama"}).json()["engine"] == "ollama"


def test_download_goes_to_the_models_engine():
    router = make_router()
    c = client(router, admin_password="pw")
    assert c.post("/admin/models/bitnet2b/download", json={"quant": "i2_s", "engine": "llamacpp"}, headers=ADMIN).status_code == 200
    assert router.adapters["llamacpp"].downloads == [("bitnet2b", "i2_s")]


def test_health_is_open_and_lists_models():
    body = client(admin_password="pw").get("/health").json()
    assert body["status"] == "ok" and "model" not in body and "bitnet2b" in body["models"]


def test_connection_settings_come_from_saved_values_and_are_validated(tmp_path):
    from pathlib import Path
    from app import build_router
    from options import SETTINGS
    from settings import Config

    store = TokenStore(str(tmp_path / "tokens.json"))
    config = Config(Path(store.settings_path), SETTINGS)
    config.set({"ollama_url": "http://box:11434", "models_dir": str(tmp_path / "gguf"), "ollama_parallel": 3})
    router = build_router(store, config)
    assert router.adapters["ollama"].url == "http://box:11434" and router.parallel == 3
    assert router.adapters["llamacpp"].models_dir == tmp_path / "gguf"
    c = TestClient(create_app(router, "pw", tokens=store, config=config, fetch_sizes=no_hub))
    assert c.put("/admin/settings", json={"values": {"ollama_url": "box:11434"}}, headers=ADMIN).status_code == 422
    assert c.put("/admin/settings", json={"values": {"ollama_url": "http://other:11434"}}, headers=ADMIN).status_code == 200


def hub_tree(_url):
    return [{"path": "lfm3-Q4_K_M.gguf", "size": 1_048_576 * 1048}, {"path": "lfm3-IQ4_XS.gguf", "size": 1_048_576 * 900},
            {"path": "lfm3-BF16.gguf", "size": 1_048_576 * 6000}]


def tight_machine(monkeypatch, router, free_mb):
    """The engines run on this machine and it has little free memory."""
    import app as app_module

    monkeypatch.setattr(app_module.hardware, "probe", lambda: {"cpu_cores": 4, "ram_total_mb": 16000, "gpu": None, "ram_available_mb": free_mb})
    for adapter in router.adapters.values():
        adapter.same_machine = True


def test_stop_start_restart_and_a_stopped_model_refuses_requests(tmp_path):
    router = make_router()
    store = TokenStore(str(tmp_path / "tokens.json"))
    c = TestClient(create_app(router, "pw", tokens=store, fetch_sizes=no_hub))
    token = {"X-Internal-Token": store.create("t")["token"]}
    act = lambda name, action: c.post(f"/admin/models/{name}/{action}", headers=ADMIN)
    assert c.post("/v1/chat", json=BODY, headers=token).status_code == 200
    assert act("lfm3", "stop").json() == {"state": "stopped"}
    refused = c.post("/v1/chat", json=BODY, headers=token)
    assert refused.status_code == 503 and "stopped" in refused.json()["detail"]
    assert c.post("/v1/chat", json={**BODY, "model": "sparkx26"}, headers=token).status_code == 200  # others keep running
    assert router.adapters["ollama"].unloads == [("lfm3", "Q4_K_M")]
    assert act("lfm3", "start").json() == {"state": "running"}
    assert c.post("/v1/chat", json=BODY, headers=token).status_code == 200
    assert act("lfm3", "restart").json() == {"state": "running"} and act("zzz", "stop").status_code == 404
    act("lfm3", "stop")
    assert next(m for m in c.get("/admin/models", headers=ADMIN).json()["models"] if m["name"] == "lfm3")["state"] == "stopped"
    from settings import load
    assert load(store.settings_path)["stopped"] == ["lfm3"]  # saved, so every replica and a restart keep it


def test_only_builds_that_fit_are_downloaded_or_loaded(monkeypatch):
    router = make_router()
    tight_machine(monkeypatch, router, 2000)  # limit 1700 MB: Q4_K_M needs 1553, Q8_0 needs 2336
    c = client(router, admin_password="pw")
    big = c.post("/admin/models/lfm3/download", json={"quant": "Q8_0", "engine": "ollama"}, headers=ADMIN)
    assert big.status_code == 409 and "Needs about" in big.json()["detail"] and router.adapters["ollama"].downloads == []
    assert c.post("/admin/models/lfm3/download", json={"quant": "Q4_K_M", "engine": "ollama"}, headers=ADMIN).status_code == 200
    assert c.put("/admin/models/lfm3/run", json={"quant": "Q8_0", "engine": "ollama"}, headers=ADMIN).status_code == 409


def test_a_request_cannot_load_a_build_that_does_not_fit(monkeypatch):
    router = make_router()
    tight_machine(monkeypatch, router, 500)
    c = client(router)
    refused = c.post("/v1/chat", json=BODY)
    assert refused.status_code == 503 and "Needs about" in refused.json()["detail"] and router.adapters["ollama"].loads == []


def test_marketplace_lists_adds_and_removes_hub_variants(tmp_path):
    router = make_router()
    store = TokenStore(str(tmp_path / "tokens.json"))
    c = TestClient(create_app(router, "pw", tokens=store, fetch_sizes=hub_tree))
    listed = next(m for m in c.get("/admin/marketplace", headers=ADMIN).json()["models"] if m["name"] == "lfm3")
    assert [v["quant"] for v in listed["variants"]] == ["IQ4_XS", "BF16"]  # Q4_K_M is curated already
    bitnet = next(m for m in c.get("/admin/marketplace", headers=ADMIN).json()["models"] if m["name"] == "bitnet2b")
    assert bitnet["variants"] == [] and "sidecar" in bitnet["note"]
    assert c.post("/admin/marketplace/lfm3/add", json={"quant": "NOPE"}, headers=ADMIN).status_code == 404
    assert c.post("/admin/marketplace/bitnet2b/add", json={"quant": "IQ4_XS"}, headers=ADMIN).status_code == 422
    assert c.post("/admin/marketplace/lfm3/add", json={"quant": "IQ4_XS"}, headers=ADMIN).status_code == 200
    lfm3 = next(m for m in c.get("/admin/models", headers=ADMIN).json()["models"] if m["name"] == "lfm3")
    assert "IQ4_XS" in [b["id"] for b in lfm3["builds"]]
    assert c.put("/admin/models/lfm3/run", json={"quant": "IQ4_XS", "engine": "ollama"}, headers=ADMIN).status_code == 200
    assert c.delete("/admin/marketplace/lfm3/IQ4_XS", headers=ADMIN).status_code == 409  # it is the build in use
    c.put("/admin/models/lfm3/run", json={"quant": "Q4_K_M", "engine": "ollama"}, headers=ADMIN)
    assert c.delete("/admin/marketplace/lfm3/IQ4_XS", headers=ADMIN).status_code == 200
    lfm3 = next(m for m in c.get("/admin/models", headers=ADMIN).json()["models"] if m["name"] == "lfm3")
    assert "IQ4_XS" not in [b["id"] for b in lfm3["builds"]]


def test_marketplace_refuses_a_variant_that_does_not_fit(monkeypatch, tmp_path):
    router = make_router()
    tight_machine(monkeypatch, router, 3000)  # limit 2550 MB: BF16 (6000 MB) cannot run
    c = TestClient(create_app(router, "pw", tokens=TokenStore(str(tmp_path / "tokens.json")), fetch_sizes=hub_tree))
    listed = next(m for m in c.get("/admin/marketplace", headers=ADMIN).json()["models"] if m["name"] == "lfm3")
    assert next(v for v in listed["variants"] if v["quant"] == "BF16")["fits"] is False
    assert c.post("/admin/marketplace/lfm3/add", json={"quant": "BF16"}, headers=ADMIN).status_code == 409


def test_progress_endpoint_and_one_download_at_a_time():
    from progress import progress

    c = client(admin_password="pw")
    assert c.get("/admin/progress", headers=ADMIN).json()["active"] is False
    progress.start("hf.co/x:Q8_0", 1000)
    try:
        progress.set(250)
        assert c.get("/admin/progress", headers=ADMIN).json() == {"active": True, "label": "hf.co/x:Q8_0", "done": 250, "total": 1000}
        busy = c.post("/admin/models/lfm3/download", json={"quant": "Q8_0", "engine": "ollama"}, headers=ADMIN)
        assert busy.status_code == 409 and "running" in busy.json()["detail"]
    finally:
        progress.finish()


def test_delete_a_downloaded_build_but_not_the_one_in_use():
    router = make_router()
    c = client(router, admin_password="pw")
    delete = lambda quant: c.post("/admin/models/lfm3/delete", json={"quant": quant, "engine": "ollama"}, headers=ADMIN)
    assert delete("Q4_K_M").status_code == 409  # the running build of a running model
    assert delete("Q1").status_code == 422 and c.post("/admin/models/zzz/delete", json={"quant": "Q4_K_M", "engine": "ollama"}, headers=ADMIN).status_code == 404
    assert delete("Q8_0").json() == {"deleted": "hf.co/LiquidAI/LFM3-1.7B-GGUF:Q8_0"}
    assert router.adapters["ollama"].deletes == [("lfm3", "Q8_0")]
    c.post("/admin/models/lfm3/stop", headers=ADMIN)
    assert delete("Q4_K_M").status_code == 200 and router.adapters["ollama"].deletes[-1] == ("lfm3", "Q4_K_M")
