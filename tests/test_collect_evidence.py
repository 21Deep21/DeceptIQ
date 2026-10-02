import json

from dataset.collect_evidence import collect_evidence


def _dns_q(domain):
    return "ok", {"a_record_count": 2, "has_mx": True, "ns_count": 2}, None


def _rdap_q(domain):
    return "not_found", None, None


def _booming_q(domain):
    raise RuntimeError("boom")


def _read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_collect_writes_logged_records(tmp_path):
    summary = collect_evidence(["a.example.com", "b.example.org"],
                               {"dns": _dns_q, "rdap": _rdap_q},
                               tmp_path, delay=0, progress_every=1)
    dns, rdap = _read(tmp_path / "dns.jsonl"), _read(tmp_path / "rdap.jsonl")
    assert len(dns) == 2 and len(rdap) == 2
    assert all(r["status"] == "ok" and r["fields"] for r in dns)
    assert all(r["status"] == "not_found" and r["fields"] is None for r in rdap)
    for r in dns + rdap:
        assert r["duration_ms"] >= 0 and r["requested_at"] and r["collector"] == "evidence/1"
    assert summary["dns"]["statuses"] == {"ok": 2}
    assert summary["rdap"]["statuses"] == {"not_found": 2}


def test_collect_resumes_without_duplicates(tmp_path):
    doms = ["a.example.com", "b.example.org", "c.example.net"]
    collect_evidence(doms, {"dns": _dns_q}, tmp_path, delay=0)
    n1 = len(_read(tmp_path / "dns.jsonl"))
    summary = collect_evidence(doms, {"dns": _dns_q}, tmp_path, delay=0)
    n2 = len(_read(tmp_path / "dns.jsonl"))
    assert n1 == 3 and n2 == 3                      # nothing re-acquired
    assert summary["dns"]["attempted"] == 0 and summary["dns"]["resumed_past"] == 3


def test_collect_limit_is_pilot_mode(tmp_path):
    summary = collect_evidence(["a.example.com", "b.example.org", "c.example.net"],
                               {"dns": _dns_q}, tmp_path, delay=0, limit=2)
    assert summary["dns"]["attempted"] == 2
    assert len(_read(tmp_path / "dns.jsonl")) == 2


def test_collect_never_dies_on_query_exception(tmp_path):
    summary = collect_evidence(["a.example.com"], {"dns": _booming_q},
                               tmp_path, delay=0)
    recs = _read(tmp_path / "dns.jsonl")
    assert len(recs) == 1 and recs[0]["status"] == "error" and recs[0]["fields"] is None
    assert summary["dns"]["statuses"] == {"error": 1}


def test_collect_retries_rate_limited_on_resume(tmp_path):
    # rate_limited is our pacing artifact -> re-attempted; the old rejection
    # stays in the log, so consumers must take the LAST record per domain
    p = tmp_path / "dns.jsonl"
    p.write_text(json.dumps({"domain": "a.example.com", "status": "rate_limited",
                             "fields": None}) + "\n", encoding="utf-8")
    summary = collect_evidence(["a.example.com"], {"dns": _dns_q}, tmp_path, delay=0)
    recs = _read(p)
    assert len(recs) == 2                       # old rejection + new attempt
    assert recs[-1]["status"] == "ok"           # last record wins
    assert summary["dns"]["attempted"] == 1
