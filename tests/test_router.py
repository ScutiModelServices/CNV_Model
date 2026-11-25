import asyncio
import pytest
from catalog import Build, ModelEntry
from errors import CapabilityError, EngineError, UnknownModel
from router import ModelRouter


class Loaded:
    def __init__(self, tag): self.tag = tag
    def generate(self, messages, temperature, max_tokens, max_time, **tools): return {"content": self.tag}
    def stream(self, messages, temperature, max_tokens, max_time, stop, **tools):
        yield "content", self.tag
        yield "done", {}


class Adapter:
    name, same_machine = "fake", True
    def __init__(self): self.loads = []
    def load(self, entry, quant): self.loads.append((entry.name, quant)); return Loaded(f"{entry.name}:{quant}")


def entry(name, caps=()):
    both = {"ollama": {}, "llamacpp": {"ref": "x.gguf", "url": "http://x:1"}}
    return ModelEntry(name, name, "o/" + name, frozenset(caps), (Build("A", 1, engines=both), Build("B", 2, engines={"ollama": {}})))


def make():
    adapter = Adapter()
    return ModelRouter({"a": entry("a", ["tools"]), "b": entry("b")}, {"ollama": adapter, "llamacpp": adapter}, {"a": "A", "b": "A"}), adapter


def run(coro): return asyncio.run(coro)


def chat(router, model="a", **extra): return run(router.chat(model, [], temperature=0, max_tokens=1, **extra))


def test_model_and_quant_in_result():
    router, _ = make()
    out = chat(router)
    assert out["content"] == "a:A" and out["model"] == "a" and out["quant"] == "A" and out["engine"] == "ollama"


def test_unknown_model_and_missing_capability():
    router, _ = make()
    with pytest.raises(UnknownModel):
        chat(router, "zzz")
    with pytest.raises(CapabilityError):
        chat(router, "b", tools=[{"type": "function"}])


def test_model_loads_once_and_set_quant_reloads():
    router, adapter = make()
    chat(router, "a"); chat(router, "a")
    assert adapter.loads == [("a", "A")]
    gen = router.generation
    router.set_choice("a", "B", "ollama")
    assert chat(router, "a")["content"] == "a:B" and router.generation > gen


def test_503_drops_the_loaded_model():
    router, adapter = make()
    class Down(Loaded):
        def generate(self, *a, **k): raise EngineError("down", 503)
    adapter.load = lambda e, q: Down("x")
    with pytest.raises(EngineError):
        chat(router, "a")
    assert not router._loaded


def test_stream_ends_with_model_in_done():
    router, _ = make()
    async def collect():
        return [event async for event in router.chat_stream("b", [], temperature=0, max_tokens=1)]
    events = run(collect())
    assert events[0] == ("content", "b:A") and events[-1][1]["model"] == "b"


def test_set_parallel_replaces_the_gate():
    router, _ = make()
    router.set_parallel(3)
    assert router.parallel == 3 and router._gate._value == 3


def test_the_user_chooses_the_engine_and_the_build_must_list_it():
    router, adapter = make()
    with pytest.raises(CapabilityError):
        chat(router, "a", engine="nope")
    assert chat(router, "a", engine="llamacpp")["engine"] == "llamacpp"  # per request
    router.set_choice("a", "A", "llamacpp")
    assert chat(router, "a")["engine"] == "llamacpp"  # saved choice
    with pytest.raises(CapabilityError):
        router.set_choice("a", "B", "llamacpp")  # build B only runs on ollama


def test_two_models_stay_loaded_at_once():
    router, adapter = make()
    chat(router, "a")
    chat(router, "b")
    assert sorted(router._loaded) == [("a", "A", "ollama"), ("b", "A", "ollama")] and router.ready


def test_stop_refuses_requests_and_start_loads_again():
    router, adapter = make()
    adapter.unload = lambda e, q: adapter.loads.append(("unloaded", e.name))
    chat(router, "a")
    assert router.state_of("a") == "running" and router.state_of("b") == "ready"
    asyncio.run(router.stop("a"))
    assert router.state_of("a") == "stopped" and ("unloaded", "a") in adapter.loads and not router._loaded
    with pytest.raises(EngineError) as stopped:
        chat(router, "a")
    assert stopped.value.status == 503 and "stopped" in str(stopped.value)
    asyncio.run(router.start("a"))
    assert router.state_of("a") == "running" and chat(router, "a")["content"] == "a:A"


def test_guard_blocks_loading_a_build_that_does_not_fit():
    router, adapter = make()
    router.guard = lambda entry, quant, engine: "Needs about 9999 MB; 1 MB of memory is free."
    with pytest.raises(EngineError) as blocked:
        chat(router, "a")
    assert blocked.value.status == 503 and "Needs about" in str(blocked.value) and adapter.loads == []
