import json
import pytest
from catalog import CatalogError, load_catalog


def write(folder, name, **over):
    body = {"name": name, "title": name.title(), "engine": "ollama", "repo": "org/" + name,
            "default_quant": "Q4_K_M", "capabilities": ["tools"],
            "builds": [{"quant": "Q4_K_M", "file_mb": 1000, "note": "usual"}, {"quant": "Q8_0", "file_mb": 1800}]}
    body.update(over)
    (folder / f"{name}.json").write_text(json.dumps(body), "utf-8")


def test_loads_entries_and_sources(tmp_path):
    write(tmp_path, "lfm3")
    entry = load_catalog(tmp_path)["lfm3"]
    assert entry.source("Q8_0") == "hf.co/org/lfm3:Q8_0"
    assert entry.build("Q4_K_M").file_mb == 1000
    assert "tools" in entry.capabilities


@pytest.mark.parametrize("over", [
    {"engine": "nope"}, {"default_quant": "Q2"}, {"capabilities": ["flying"]}, {"builds": []}])
def test_rejects_bad_entry(tmp_path, over):
    write(tmp_path, "x", **over)
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)


def test_url_only_for_llamacpp_and_required_there(tmp_path):
    write(tmp_path, "a", url="http://x:1")
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)
    (tmp_path / "a.json").unlink()
    write(tmp_path, "b", engine="llamacpp", builds=[{"quant": "Q4_K_M", "file_mb": 1, "ref": "b.gguf"}])
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)
    (tmp_path / "b.json").unlink()
    write(tmp_path, "c", engine="llamacpp", url="http://x:1", builds=[{"quant": "Q4_K_M", "file_mb": 1, "ref": "c.gguf"}])
    assert load_catalog(tmp_path)["c"].url == "http://x:1"


def test_name_must_match_file(tmp_path):
    write(tmp_path, "a")
    (tmp_path / "a.json").rename(tmp_path / "b.json")
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)


def test_shipped_catalog_loads():
    from pathlib import Path
    names = set(load_catalog(Path(__file__).parent.parent / "configs" / "models"))
    assert names == {"lfm3", "sparkx26", "minicpm52b", "bitnet2b"}
