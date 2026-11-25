from pathlib import Path

import pytest

from catalog import load_catalog
from router import ModelRouter

CATALOG = load_catalog(Path(__file__).parent.parent / "configs" / "models")


@pytest.fixture(autouse=True)
def _tmp_tokens_file(tmp_path, monkeypatch):
    """Keeps tokens.json and settings.json out of the repo, and a saved setting from leaking into the next test."""
    monkeypatch.setenv("SERVICE_TOKENS_FILE", str(tmp_path / "tokens.json"))


@pytest.fixture(autouse=True)
def _cpu_with_avx2(monkeypatch):
    """The machine the tests run on does not matter: it has AVX2 unless a test says otherwise."""
    import builds

    monkeypatch.setattr(builds, "cpu_flags", lambda: {"avx", "avx2"})


class FakeLoaded:
    """Stands in for a loaded model: answers `reply` and records every call."""

    def __init__(self, source, reply="hello", fail=None, tool_calls=None):
        self.source, self.reply, self.calls, self.tool_calls, self.fail = source, reply, [], tool_calls, fail

    def generate(self, messages, temperature, max_tokens, max_time, tools=None):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **({"tools": tools} if tools else {})})
        if self.fail:
            raise self.fail
        return {"content": self.reply, "prompt_tokens": 3, "completion_tokens": 5, "finish": "stop", **({"tool_calls": self.tool_calls} if self.tool_calls else {})}

    def stream(self, messages, temperature, max_tokens, max_time, stop, tools=None):
        if self.fail:
            raise self.fail
        words = self.reply.split(" ")
        for number, word in enumerate(words):
            yield "content", word + (" " if number < len(words) - 1 else "")
        yield "done", {"prompt_tokens": 3, "completion_tokens": 5, "finish": "stop"}


class FakeAdapter:
    """Both engines in one fake: `loads` lists (model, quant) pairs, `downloads` what the admin fetched."""

    same_machine = False

    def __init__(self, name, **options):
        self.name, self.options, self.loads, self.downloads, self.unloads, self.deletes = name, options, [], [], [], []

    def load(self, entry, quant):
        self.loads.append((entry.name, quant))
        return FakeLoaded(entry.source(quant, self.name), **self.options)

    def unload(self, entry, quant):
        self.unloads.append((entry.name, quant))

    def delete(self, entry, quant):
        self.deletes.append((entry.name, quant))

    def downloaded(self):
        """Every build counts as downloaded, so a test can select any of them."""
        return {e.source(b.quant, self.name) for e in CATALOG.values() for b in e.builds if self.name in b.engines}

    def download(self, entry, quant):
        self.downloads.append((entry.name, quant))


def make_router(**options):
    """A router over the shipped catalog whose engines are fakes."""
    adapters = {"ollama": FakeAdapter("ollama", **options), "llamacpp": FakeAdapter("llamacpp", **options)}
    return ModelRouter(dict(CATALOG), adapters, {name: item.builds[0].quant for name, item in CATALOG.items()})  # nothing is selected by default; the tests select the first build
