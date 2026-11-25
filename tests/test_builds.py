from builds import cached_sizes, hub_sizes, listing
from catalog import Build, ModelEntry

ENGINES = {"ollama": {}, "llamacpp": {"ref": "m.gguf", "url": "http://x:1"}}
ENTRY = ModelEntry("m", "M", "org/m", "Q4_K_M", frozenset(),
                   (Build("Q4_K_M", 1000, engines={"ollama": {}}), Build("Q8_0", 1800, engines=ENGINES), Build("F16", 3300, engines={"ollama": {}})))
HW = {"ollama": {"gpu": None, "ram_available_mb": 3000}, "llamacpp": {"gpu": None, "ram_available_mb": 3000}}  # CPU limit 2550 MB: Q8_0 needs 2380, F16 needs 4030
CURRENT = ("Q4_K_M", "ollama")


def option(rows, quant, engine):
    return next(o for r in rows if r["id"] == quant for o in r["engines"] if o["engine"] == engine)


def test_listing_marks_fit_current_downloaded_and_sorts():
    rows = listing(ENTRY, HW, CURRENT, {"ollama": {"hf.co/org/m:Q4_K_M"}}, {})
    assert [r["id"] for r in rows] == ["Q4_K_M", "Q8_0", "F16"]
    assert rows[0]["current"] and option(rows, "Q4_K_M", "ollama") == {
        "engine": "ollama", "current": True, "fits": True, "reason": "", "runs_on": "cpu", "downloaded": True}
    assert option(rows, "Q8_0", "ollama")["fits"] is True and not option(rows, "Q8_0", "ollama")["downloaded"]
    assert option(rows, "F16", "ollama")["fits"] is False and "Needs about" in option(rows, "F16", "ollama")["reason"]


def test_each_build_lists_only_the_engines_that_can_run_it():
    rows = listing(ENTRY, HW, CURRENT, {}, {})
    assert [o["engine"] for o in rows[1]["engines"]] == ["ollama", "llamacpp"] and len(rows[0]["engines"]) == 1


def test_downloaded_is_per_engine():
    rows = listing(ENTRY, HW, CURRENT, {"llamacpp": {"org/m/m.gguf"}}, {})
    assert option(rows, "Q8_0", "llamacpp")["downloaded"] and not option(rows, "Q8_0", "ollama")["downloaded"]


def test_listing_unknown_hardware_is_not_a_no():
    rows = listing(ENTRY, {"ollama": None, "llamacpp": None}, CURRENT, {}, {})
    assert option(rows, "Q8_0", "ollama")["fits"] is None


def test_live_size_replaces_stored_size():
    assert listing(ENTRY, HW, CURRENT, {}, {"Q4_K_M": 1111})[0]["disk_mb"] == 1111


def fake_fetch(_url):
    return [{"path": "m-Q4_K_M.gguf", "size": 1_048_576 * 1010}, {"path": "README.md", "size": 5},
            {"path": "m-q8_0.gguf", "size": 1_048_576 * 1790}]


def test_hub_sizes_reads_gguf_files():
    assert hub_sizes("org/m", fake_fetch) == {"Q4_K_M": 1010, "Q8_0": 1790}


def test_cached_sizes_uses_cache_then_falls_back(tmp_path):
    path = tmp_path / "sizes.json"
    assert cached_sizes(path, "org/m", fake_fetch, now=1000)["Q4_K_M"] == 1010

    def boom(_url):
        raise OSError("offline")

    assert cached_sizes(path, "org/m", boom, now=1500)["Q4_K_M"] == 1010   # fresh cache, no fetch
    assert cached_sizes(path, "org/m", boom, now=999999)["Q4_K_M"] == 1010  # stale but Hub down
    assert cached_sizes(tmp_path / "none.json", "org/m", boom, now=1) == {}
