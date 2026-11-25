import threading
from catalog import Build, ModelEntry
from adapters.llamacpp import LlamaCppAdapter
from adapters.ollama import OllamaAdapter

ENTRY = ModelEntry("bitnet2b", "BitNet", "org/bitnet", "i2_s", frozenset(),
                   (Build("i2_s", 1200, engines={"llamacpp": {"ref": "bitnet-i2_s.gguf", "url": "http://llama:8080"}}),))
REPLY = {"choices": [{"message": {"content": " hi "}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def adapter_with_file(tmp_path):
    path = tmp_path / "org" / "bitnet"
    path.mkdir(parents=True)
    (path / "bitnet-i2_s.gguf").write_bytes(b"x")
    return LlamaCppAdapter(tmp_path)


def test_llamacpp_generate_maps_openai_reply(tmp_path, monkeypatch):
    adapter, calls = adapter_with_file(tmp_path), []
    monkeypatch.setattr(adapter, "_post", lambda url, path, body, timeout: (calls.append(url), REPLY)[1])
    out = adapter.load(ENTRY, "i2_s").generate([{"role": "user", "content": "yo"}], 0.2, 16, 5.0)
    assert out == {"content": "hi", "prompt_tokens": 3, "completion_tokens": 1, "finish": "stop"}
    assert calls == ["http://llama:8080"]  # the model's own server, from the catalog


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
