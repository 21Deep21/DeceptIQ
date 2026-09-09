from dataset.build_dataset import apply_domain_cap, clean_entries, sample_balanced


def _e(url, label, source="openphish", threat_type="phishing"):
    return {"url": url, "label": label, "source": source, "threat_type": threat_type}


def test_clean_normalizes_dedups_and_drops_malformed():
    entries = [
        _e("https://evil.example.com/login", 1),
        _e("  https://evil.example.com/login  ", 1),
        _e("ht!tp://bad url", 1),
        _e("https://good.example.com/", 0, source="tranco", threat_type="benign"),
        _e("https://good.example.com/", 0, source="tranco", threat_type="benign"),
    ]
    cleaned, stats = clean_entries(entries)
    urls = [e["url"] for e in cleaned]
    assert urls.count("https://evil.example.com/login") == 1
    assert urls.count("https://good.example.com/") == 1
    assert stats["input"] == 5
    assert stats["malformed_removed"] == 1
    assert stats["duplicates_removed"] == 2
    assert stats["after_cleaning"] == 2


def test_cross_label_duplicate_keeps_malicious():
    entries = [
        _e("https://x.example.com/", 0, source="tranco", threat_type="benign"),
        _e("https://x.example.com/", 1),
    ]
    cleaned, stats = clean_entries(entries)
    assert stats["label_conflicts_resolved"] == 1
    assert cleaned[0]["label"] == 1

    entries.reverse()
    cleaned, stats = clean_entries(entries)
    assert stats["label_conflicts_resolved"] == 1
    assert cleaned[0]["label"] == 1


def test_clean_extracts_registrable_domain():
    cleaned, _ = clean_entries([_e("https://a.b.example.co.uk/login", 1)])
    assert cleaned[0]["registrable_domain"] == "example.co.uk"


def test_clean_drops_hosts_without_registrable_domain():
    cleaned, stats = clean_entries([_e("http://host.example/page", 1)])
    assert cleaned == []
    assert stats["malformed_removed"] == 1


def test_domain_cap_is_per_label():
    mal = [_e(f"https://evil.example.com/p{i}", 1) for i in range(4)]
    ben = [_e(f"https://docs.example.org/g{i}", 0, source="sitemap", threat_type="benign")
           for i in range(5)]
    entries, _ = clean_entries(mal + ben)
    kept, dropped = apply_domain_cap(entries, benign_cap=3, malicious_cap=2)
    assert len([e for e in kept if e["label"] == 1]) == 2
    assert len([e for e in kept if e["label"] == 0]) == 3
    assert dropped == 4


def test_sample_balanced_benign_mix():
    obs = [_e(f"https://s{i}.example.com/page{i}", 0, source="sitemap", threat_type="benign")
           for i in range(20)]
    const = [_e(f"https://c{i}.example.com/", 0, source="tranco", threat_type="benign")
             for i in range(20)]
    mal = [_e(f"https://m{i}.example.net/x", 1) for i in range(20)]
    sel, stats = sample_balanced(mal + obs + const, 10, 10, seed=3, benign_observed_frac=0.5)
    assert stats["malicious_selected"] == 10
    assert stats["benign_observed_selected"] == 5
    assert stats["benign_constructed_selected"] == 5
    sources = {e["source"] for e in sel if e["label"] == 0}
    assert sources == {"sitemap", "tranco"}


def test_sample_balanced_mix_tops_up_when_observed_short():
    obs = [_e("https://s.example.com/p", 0, source="sitemap", threat_type="benign")]
    const = [_e(f"https://c{i}.example.com/", 0, source="tranco", threat_type="benign")
             for i in range(20)]
    sel, stats = sample_balanced(obs + const, 0, 10, seed=1, benign_observed_frac=0.5)
    # topped up with REAL constructed benign data - never fabricated
    assert stats["benign_observed_selected"] == 1
    assert stats["benign_constructed_selected"] == 9
    assert stats["benign_selected"] == 10


def test_sample_balanced_is_seed_deterministic():
    entries = (
        [_e(f"https://m{i}.example.com/x", 1) for i in range(30)]
        + [_e(f"https://l{i}.example.com/", 0, source="tranco", threat_type="benign")
           for i in range(30)]
    )
    a, _ = sample_balanced(entries, 10, 10, seed=7, benign_observed_frac=0.5)
    b, _ = sample_balanced(entries, 10, 10, seed=7, benign_observed_frac=0.5)
    assert [e["url"] for e in a] == [e["url"] for e in b]


def test_sample_balanced_never_fabricates():
    entries = [_e("https://only.example.com/x", 1)]
    result, stats = sample_balanced(entries, 10, 10, seed=1, benign_observed_frac=0.5)
    assert stats["malicious_selected"] == 1
    assert stats["benign_selected"] == 0
    assert len(result) == 1


from dataset.build_dataset import load_openphish_snapshots


def test_load_openphish_snapshots_merges(tmp_path):
    (tmp_path / "openphish_feed.txt").write_text(
        "https://a.example.com/1\n", encoding="utf-8")
    (tmp_path / "openphish_20260901.txt").write_text(
        "https://a.example.com/1\nhttps://b.example.com/2\n", encoding="utf-8")
    entries, names = load_openphish_snapshots(tmp_path)
    assert len(entries) == 3   # duplicates removed LATER by clean_entries
    assert set(names) == {"openphish_20260901.txt", "openphish_feed.txt"}
    assert all(e["label"] == 1 and e["source"] == "openphish" for e in entries)


def test_load_openphish_snapshots_empty(tmp_path):
    entries, names = load_openphish_snapshots(tmp_path)
    assert entries == [] and names == []
