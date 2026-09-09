import pandas as pd

from model.split import domain_aware_split


def _make_df(n_dom=12, per_domain=3):
    rows = []
    for i in range(n_dom):
        for j in range(per_domain):
            rows.append(dict(url=f"https://m{i}.example.com/p{j}", label=1,
                             source="s", threat_type="phishing",
                             registrable_domain=f"m{i}.example.com"))
            rows.append(dict(url=f"https://b{i}.example.org/", label=0,
                             source="t", threat_type="benign",
                             registrable_domain=f"b{i}.example.org"))
    return pd.DataFrame(rows)


def test_split_domain_purity():
    df = _make_df()
    train, val, test, stats = domain_aware_split(df, seed=42)
    d_tr, d_va, d_te = (set(x["registrable_domain"]) for x in (train, val, test))
    assert not (d_tr & d_va) and not (d_tr & d_te) and not (d_va & d_te)
    assert all(v == 0 for v in stats["domain_overlap"].values())
    assert len(train) + len(val) + len(test) == len(df)


def test_split_contains_both_classes():
    df = _make_df()
    train, val, test, _ = domain_aware_split(df, seed=42)
    for part in (train, val, test):
        assert set(part["label"]) == {0, 1}


def test_split_deterministic():
    df = _make_df()
    a = domain_aware_split(df, seed=7)
    b = domain_aware_split(df, seed=7)
    assert list(a[0]["url"]) == list(b[0]["url"])
    assert list(a[1]["url"]) == list(b[1]["url"])


def test_dual_label_domain_stays_in_one_split():
    df = _make_df(n_dom=6)
    extra = pd.DataFrame(
        [dict(url=f"https://m0.example.com/benign{j}", label=0, source="t",
              threat_type="benign", registrable_domain="m0.example.com") for j in range(2)]
    )
    df = pd.concat([df, extra], ignore_index=True)
    train, val, test, stats = domain_aware_split(df, seed=42)
    counts = [(p["registrable_domain"] == "m0.example.com").sum() for p in (train, val, test)]
    assert 5 in counts                       # 3 malicious + 2 benign rows together
    assert sum(1 for c in counts if c > 0) == 1  # all in exactly ONE split
    assert stats["dual_label_domains"] == 1


def test_dataset_fingerprint_detects_content_change():
    from model.split import dataset_fingerprint
    df = _make_df()
    fp = dataset_fingerprint(df)
    assert fp == dataset_fingerprint(df.copy())            # stable
    df2 = df.copy()
    df2.loc[0, "url"] = "https://different.example.com/x"  # same row count
    assert dataset_fingerprint(df2) != fp                   # content change detected
