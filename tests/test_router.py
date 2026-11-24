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
    return ModelEntry(name, name, "ollama", "o/" + name, "A", frozenset(caps), (Build("A", 1), Build("B", 2)))


def make():
    adapter = Adapter()
    return ModelRouter({"a": entry("a", ["tools"]), "b": entry("b")}, {"ollama": adapter}), adapter


def run(coro): return asyncio.run(coro)


def chat(router, model="a", **extra): return run(router.chat(model, [], temperature=0, max_tokens=1, **extra))


def test_model_and_quant_in_result():
    router, _ = make()
    out = chat(router)
    assert out["content"] == "a:A" and out["model"] == "a" and out["quant"] == "A"


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
    router.set_quant("a", "B")
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
