import json
import pytest
from catalog import CatalogError, load_catalog


def write(folder, name, **over):
    body = {"name": name, "title": name.title(), "repo": "org/" + name,
            "default_quant": "Q4_K_M", "capabilities": ["tools"],
            "builds": [{"quant": "Q4_K_M", "file_mb": 1000, "note": "usual", "engines": {"ollama": {}}},
                       {"quant": "Q8_0", "file_mb": 1800, "engines": {"ollama": {}, "llamacpp": {"ref": "a-Q8_0.gguf", "url": "http://x:1"}}}]}
    body.update(over)
    (folder / f"{name}.json").write_text(json.dumps(body), "utf-8")


def test_loads_entries_and_sources(tmp_path):
    write(tmp_path, "lfm3")
    entry = load_catalog(tmp_path)["lfm3"]
    assert entry.source("Q8_0", "ollama") == "hf.co/org/lfm3:Q8_0"
    assert entry.source("Q8_0", "llamacpp") == "org/lfm3/a-Q8_0.gguf" and entry.url("Q8_0", "llamacpp") == "http://x:1"
    assert entry.engines("Q4_K_M") == ("ollama",) and entry.engines("Q8_0") == ("ollama", "llamacpp")
    assert entry.build("Q4_K_M").file_mb == 1000
    assert "tools" in entry.capabilities


@pytest.mark.parametrize("over", [
    {"default_quant": "Q2"}, {"capabilities": ["flying"]}, {"builds": []},
    {"builds": [{"quant": "Q4_K_M", "file_mb": 1, "engines": {}}]},
    {"builds": [{"quant": "Q4_K_M", "file_mb": 1, "engines": {"nope": {}}}]},
    {"builds": [{"quant": "Q4_K_M", "file_mb": 1, "engines": {"llamacpp": {"ref": "x.gguf"}}}]},
    {"builds": [{"quant": "Q4_K_M", "file_mb": 1, "engines": {"ollama": {"url": "http://x:1"}}}]}])
def test_rejects_bad_entry(tmp_path, over):
    write(tmp_path, "x", **over)
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)


def test_llamacpp_prompt_option_is_optional_and_checked(tmp_path):
    def with_options(options):
        write(tmp_path, "p", engine=None, builds=[{"quant": "Q4_K_M", "file_mb": 1, "engines": {"llamacpp": options}}])
        return load_catalog(tmp_path)

    base = {"ref": "p.gguf", "url": "http://x:1"}
    assert with_options({**base, "prompt": "bitnet"})["p"].options("Q4_K_M", "llamacpp")["prompt"] == "bitnet"
    assert "prompt" not in with_options(base)["p"].options("Q4_K_M", "llamacpp")
    with pytest.raises(CatalogError):
        with_options({**base, "prompt": "nope"})


def test_name_must_match_file(tmp_path):
    write(tmp_path, "a")
    (tmp_path / "a.json").rename(tmp_path / "b.json")
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)


def test_shipped_catalog_loads():
    from pathlib import Path
    names = set(load_catalog(Path(__file__).parent.parent / "configs" / "models"))
    assert names == {"lfm3", "sparkx26", "minicpm52b", "bitnet2b"}
