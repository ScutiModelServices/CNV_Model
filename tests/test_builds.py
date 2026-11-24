from builds import cached_sizes, hub_sizes, listing
from catalog import Build, ModelEntry

ENTRY = ModelEntry("m", "M", "ollama", "org/m", "Q4_K_M", frozenset(),
                   (Build("Q4_K_M", 1000), Build("Q8_0", 1800), Build("F16", 3300)))
HW = {"gpu": None, "ram_available_mb": 3000}  # CPU limit 2550 MB: Q8_0 needs 2380, F16 needs 4030


def test_listing_marks_fit_current_downloaded_and_sorts():
    rows = listing(ENTRY, HW, "Q4_K_M", {"hf.co/org/m:Q4_K_M"}, {})
    assert [r["id"] for r in rows] == ["Q4_K_M", "Q8_0", "F16"]
    assert rows[0]["current"] and rows[0]["downloaded"] and rows[0]["fits"]
    assert rows[1]["fits"] is True
    assert rows[2]["fits"] is False and "Needs about" in rows[2]["reason"]


def test_listing_unknown_hardware_is_not_a_no():
    rows = listing(ENTRY, None, "Q4_K_M", set(), {})
    assert all(r["fits"] is None for r in rows[1:])


def test_live_size_replaces_stored_size():
    assert listing(ENTRY, HW, "Q4_K_M", set(), {"Q4_K_M": 1111})[0]["disk_mb"] == 1111


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
