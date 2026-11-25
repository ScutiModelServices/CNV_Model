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


def test_marketplace_lists_hub_quants_the_catalog_lacks_and_whether_they_fit():
    from builds import marketplace
    files = {"Q4_K_M": {"path": "m-Q4_K_M.gguf", "mb": 1000}, "IQ4_XS": {"path": "m-IQ4_XS.gguf", "mb": 900},
             "BF16": {"path": "m-BF16.gguf", "mb": 6000}}
    rows = marketplace(ENTRY, {"Q4_K_M"}, files, HW["ollama"], {"IQ4_XS"})
    assert [r["quant"] for r in rows] == ["IQ4_XS", "BF16"]
    assert rows[0]["fits"] is True and rows[0]["added"] and rows[1]["fits"] is False and not rows[1]["added"]


def test_hub_files_keeps_file_names():
    from builds import hub_files
    assert hub_files("org/m", fake_fetch)["Q8_0"] == {"path": "m-q8_0.gguf", "mb": 1790}


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


def test_a_build_that_needs_avx2_does_not_fit_a_cpu_without_it():
    needs = ModelEntry("b", "B", "org/b", "i2_s", frozenset(),
                       (Build("i2_s", 1000, engines={"llamacpp": {"ref": "b.gguf", "url": "http://x:1", "cpu": "avx2"}}),))
    hw = {"llamacpp": {"gpu": None, "ram_available_mb": 8000}}
    old = listing(needs, hw, ("i2_s", "llamacpp"), {}, {}, {"sse4_2", "avx"})[0]["engines"][0]
    assert old["fits"] is False and "AVX2" in old["reason"]  # even though it is the running build
    assert listing(needs, hw, ("i2_s", "llamacpp"), {}, {}, {"avx", "avx2"})[0]["engines"][0]["fits"] is True
    assert listing(needs, hw, ("i2_s", "llamacpp"), {}, {}, None)[0]["engines"][0]["fits"] is True  # flags unreadable: allowed
