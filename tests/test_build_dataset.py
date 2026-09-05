from dataset.build_dataset import apply_domain_cap, clean_entries, sample_balanced


def _e(url, label, source="openphish", threat_type="phishing"):
    return {"url": url, "label": label, "source": source, "threat_type": threat_type}


def test_clean_normalizes_dedups_and_drops_malformed():
    entries = [
        _e("https://evil.example.com/login", 1),
        _e("  https://evil.example.com/login  ", 1),  # duplicate after normalization
        _e("ht!tp://bad url", 1),  # malformed
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
    # '.example' is a reserved TLD name but NOT in the Public Suffix List,
    # so no registrable domain exists and the row is dropped by design.
    # The live feed run dropped 31 rows for the same reason (bare public
    # suffixes such as niigata.jp, act.edu.au) - this locks that behavior.
    cleaned, stats = clean_entries([_e("http://host.example/page", 1)])
    assert cleaned == []
    assert stats["malformed_removed"] == 1


def test_domain_cap_limits_per_domain():
    # NOTE: evil.example.com and other.example.com would SHARE the
    # registrable domain example.com - use .com/.org for distinct groups.
    # apply_domain_cap operates on POST-CLEANING records (contract:
    # records carry registrable_domain).
    entries = [_e(f"https://evil.example.com/p{i}", 1) for i in range(6)]
    entries += [_e(f"https://evil.example.org/p{i}", 1) for i in range(2)]
    cleaned, _ = clean_entries(entries)
    assert len(cleaned) == 8
    kept, dropped = apply_domain_cap(cleaned, cap=2)
    assert len(kept) == 4  # 2 from each registrable domain
    assert dropped == 4


def test_sample_balanced_is_seed_deterministic():
    entries = (
        [_e(f"https://m{i}.example.com/x", 1) for i in range(30)]
        + [_e(f"https://l{i}.example.com/", 0, source="tranco", threat_type="benign") for i in range(30)]
    )
    a, sa = sample_balanced(entries, 10, 10, seed=7)
    b, _ = sample_balanced(entries, 10, 10, seed=7)
    assert sa["malicious_selected"] == 10 and sa["benign_selected"] == 10
    assert [e["url"] for e in a] == [e["url"] for e in b]


def test_sample_balanced_never_fabricates():
    entries = [_e("https://only.example.com/x", 1)]
    result, stats = sample_balanced(entries, 10, 10, seed=1)
    assert stats["malicious_selected"] == 1
    assert stats["benign_selected"] == 0
    assert len(result) == 1
