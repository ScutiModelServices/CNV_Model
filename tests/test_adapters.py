import json
import threading
from catalog import Build, ModelEntry
from adapters.llamacpp import LlamaCppAdapter
from adapters.ollama import OllamaAdapter

ENTRY = ModelEntry("bitnet2b", "BitNet", "org/bitnet", frozenset(),
                   (Build("i2_s", 1200, engines={"llamacpp": {"ref": "bitnet-i2_s.gguf", "url": "http://llama:8080"}}),))
REPLY = {"choices": [{"message": {"content": " hi "}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def adapter_with_file(tmp_path):
    path = tmp_path / "org" / "bitnet"
    path.mkdir(parents=True)
    (path / "bitnet-i2_s.gguf").write_bytes(b"x")
    return LlamaCppAdapter(tmp_path)


def fake_open(adapter, monkeypatch, replies):
    """Replace the HTTP call: `replies` maps a path to the JSON the server answers; returns the (url, path, body) calls."""
    import io

    calls = []

    def _open(url, path, body, timeout):
        calls.append((url, path, body))
        return io.BytesIO(json.dumps(replies[path]).encode())

    monkeypatch.setattr(adapter, "_open", _open)
    return calls


def test_llamacpp_generate_maps_openai_reply(tmp_path, monkeypatch):
    adapter = adapter_with_file(tmp_path)
    calls = fake_open(adapter, monkeypatch, {"/v1/chat/completions": REPLY})
    out = adapter.load(ENTRY, "i2_s").generate([{"role": "user", "content": "yo"}], 0.2, 16, 5.0)
    assert out == {"content": "hi", "prompt_tokens": 3, "completion_tokens": 1, "finish": "stop"}
    assert calls[0][0] == "http://llama:8080" and calls[0][1] == "/v1/chat/completions"  # the model's own server, from the catalog


def test_llamacpp_prompt_bitnet_renders_the_prompt_and_uses_completion(tmp_path, monkeypatch):
    adapter = adapter_with_file(tmp_path)
    entry = ModelEntry("bitnet2b", "BitNet", "org/bitnet", frozenset(),
                       (Build("i2_s", 1200, engines={"llamacpp": {"ref": "bitnet-i2_s.gguf", "url": "http://llama:8080", "prompt": "bitnet"}}),))
    calls = fake_open(adapter, monkeypatch, {"/tokenize": {"tokens": [1, 2, 3]},
                                             "/completion": {"content": " hi ", "tokens_evaluated": 3, "tokens_predicted": 1, "stopped_limit": False}})
    out = adapter.load(entry, "i2_s").generate([{"role": "user", "content": " yo "}], 0.2, 16, 5.0)
    assert out == {"content": "hi", "prompt_tokens": 3, "completion_tokens": 1, "finish": "stop"}
    assert [c[1] for c in calls] == ["/tokenize", "/completion"]
    assert calls[1][2]["prompt"] == "User: yo<|eot_id|>Assistant: " and calls[1][2]["stop"] == ["<|eot_id|>"] and calls[1][2]["n_predict"] == 16


def test_llamacpp_prompt_bitnet_refuses_a_conversation_longer_than_the_context(tmp_path, monkeypatch):
    import pytest
    from errors import EngineError

    adapter = adapter_with_file(tmp_path)
    entry = ModelEntry("bitnet2b", "BitNet", "org/bitnet", frozenset(),
                       (Build("i2_s", 1200, engines={"llamacpp": {"ref": "bitnet-i2_s.gguf", "url": "http://llama:8080", "prompt": "bitnet"}}),))
    fake_open(adapter, monkeypatch, {"/tokenize": {"tokens": list(range(4096))}})
    with pytest.raises(EngineError) as long:
        adapter.load(entry, "i2_s").generate([{"role": "user", "content": "x"}], 0, 8, 5.0)
    assert long.value.status == 400


def test_llamacpp_stream_reads_sse(tmp_path, monkeypatch):
    adapter = adapter_with_file(tmp_path)
    lines = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n', b'data: {"choices":[{"delta":{"content":"b"},"finish_reason":"length"}]}\n', b"data: [DONE]\n"]

    class Response(list):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(adapter, "_open", lambda url, path, body, timeout: Response(lines))
    events = list(adapter.load(ENTRY, "i2_s").stream([], 0, 8, 5.0, threading.Event()))
    assert events[:2] == [("content", "a"), ("content", "b")] and events[-1][1]["finish"] == "length"


def test_llamacpp_downloaded_and_missing_file(tmp_path):
    adapter = adapter_with_file(tmp_path)
    assert adapter.downloaded() == {"org/bitnet/bitnet-i2_s.gguf"}
    assert adapter.local_ref(ENTRY, "i2_s") == "1"
    empty = LlamaCppAdapter(tmp_path / "none")
    assert empty.downloaded() == set()


def test_ollama_downloaded_strips_latest(monkeypatch):
    adapter = OllamaAdapter("http://127.0.0.1:11434")
    monkeypatch.setattr("adapters.ollama._json_call", lambda *a, **k: {"models": [{"name": "hf.co/o/m:Q4_K_M"}, {"name": "x:latest"}]})
    assert adapter.downloaded() == {"hf.co/o/m:Q4_K_M", "x"} and adapter.same_machine


def test_llamacpp_delete_removes_the_file_and_a_half_download(tmp_path):
    adapter = adapter_with_file(tmp_path)
    part = tmp_path / "org" / "bitnet" / "bitnet-i2_s.gguf.part"
    part.write_bytes(b"x")
    adapter.delete(ENTRY, "i2_s")
    assert adapter.downloaded() == set() and not part.exists()
    adapter.delete(ENTRY, "i2_s")  # already gone is fine


def test_ollama_delete_sends_delete_and_ignores_not_found(monkeypatch):
    import urllib.error

    sent = []

    def fake_urlopen(request, timeout=0):
        sent.append((request.get_method(), request.full_url, request.data))
        raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)

    monkeypatch.setattr("adapters.ollama.urllib.request.urlopen", fake_urlopen)
    entry = ModelEntry("m", "M", "org/m", frozenset(), (Build("Q4_K_M", 1, engines={"ollama": {}}),))
    OllamaAdapter("http://127.0.0.1:11434").delete(entry, "Q4_K_M")
    assert sent == [("DELETE", "http://127.0.0.1:11434/api/delete", b'{"model": "hf.co/org/m:Q4_K_M"}')]


def test_ollama_download_rebuilds_a_model_that_names_a_renderer(monkeypatch):
    import contextlib

    calls = []
    monkeypatch.setattr("adapters.ollama._call", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr("progress.ollama_pull_progress", lambda response, model: "success")
    monkeypatch.setattr("adapters.ollama._json_call", lambda url, path, body, timeout, model: calls.append((path, body)) or {"status": "success"})
    monkeypatch.setattr(OllamaAdapter, "_manifest", staticmethod(lambda name: (b"", {"layers": [
        {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:p"},
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:m"}]})))
    plain = ModelEntry("m", "M", "org/m", frozenset(), (Build("Q4", 1, engines={"ollama": {}}),
                                                         Build("Q2", 1, engines={"ollama": {"renderer": "r", "parser": "r"}})))
    adapter = OllamaAdapter("http://127.0.0.1:11434")
    adapter.download(plain, "Q4")
    assert calls == []  # an ordinary build is only pulled
    adapter.download(plain, "Q2")
    assert calls == [("/api/create", {"model": "hf.co/org/m:Q2", "files": {"model.gguf": "sha256:m"}, "stream": False, "renderer": "r", "parser": "r"})]
