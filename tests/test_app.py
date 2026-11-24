import pytest
from fastapi.testclient import TestClient

from app import create_app
from conftest import make_router
from errors import EngineError
from tokens import TokenStore

BODY = {"messages": [{"role": "user", "content": "hi"}]}
TOOL = {"name": "f", "parameters": {"type": "object", "properties": {}}}
ADMIN = {"X-Internal-Token": "pw"}


def no_hub(_url):
    raise OSError("offline")


def client(router=None, admin_password=None, **options):
    return TestClient(create_app(router or make_router(), admin_password, fetch_sizes=no_hub, **options))


def test_chat_defaults_to_default_model_and_reports_usage():
    body = client().post("/v1/chat", json={**BODY, "temperature": 0.3, "max_tokens": 50}).json()
    assert body["content"] == "hello" and body["finish_reason"] == "stop"
    assert body["model"] == "lfm3" and body["quant"] == "Q4_K_M"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 5} and "duration_ms" in body


def test_chat_picks_the_named_model_and_engine():
    router = make_router()
    body = client(router).post("/v1/chat", json={**BODY, "model": "bitnet2b"}).json()
    assert body["model"] == "bitnet2b" and router.adapters["llamacpp"].loads == [("bitnet2b", "i2_s")]


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


def test_models_lists_capabilities():
    rows = client().get("/v1/models").json()
    lfm3 = next(r for r in rows if r["name"] == "lfm3")
    assert lfm3["default"] and "tools" in lfm3["capabilities"] and {r["name"] for r in rows} >= {"bitnet2b", "minicpm52b", "sparkx26"}


def test_admin_models_lists_builds_under_each_model():
    c = client(admin_password="pw")
    assert c.get("/admin/models").status_code == 401
    rows = c.get("/admin/models", headers=ADMIN).json()["models"]
    lfm3 = next(r for r in rows if r["name"] == "lfm3")
    assert [b["id"] for b in lfm3["builds"]][0] == "Q4_K_M" and all("downloaded" in b for b in lfm3["builds"])
    assert lfm3["current_quant"] == "Q4_K_M"


def test_set_quant_saves_and_applies(tmp_path):
    router = make_router()
    store = TokenStore(str(tmp_path / "tokens.json"))
    c = TestClient(create_app(router, "pw", tokens=store, fetch_sizes=no_hub))
    assert c.put("/admin/models/lfm3/quant", json={"quant": "Q1"}, headers=ADMIN).status_code == 422
    assert c.put("/admin/models/zzz/quant", json={"quant": "Q4_K_M"}, headers=ADMIN).status_code == 404
    assert c.put("/admin/models/lfm3/quant", json={"quant": "Q8_0"}, headers=ADMIN).status_code == 200
    token = {"X-Internal-Token": store.create("t")["token"]}
    assert c.post("/v1/chat", json=BODY, headers=token).json()["quant"] == "Q8_0"


def test_download_goes_to_the_models_engine():
    router = make_router()
    c = client(router, admin_password="pw")
    assert c.post("/admin/models/bitnet2b/download", json={"quant": "i2_s"}, headers=ADMIN).status_code == 200
    assert router.adapters["llamacpp"].downloads == [("bitnet2b", "i2_s")]


def test_health_is_open_and_lists_models():
    body = client(admin_password="pw").get("/health").json()
    assert body["status"] == "ok" and body["model"] == "lfm3" and "bitnet2b" in body["models"]
