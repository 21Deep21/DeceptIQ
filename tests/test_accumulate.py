import datetime as dt
from pathlib import Path

import requests

from dataset.accumulate_feeds import accumulate, snapshot_path

FEED = b"https://phish1.example.com/login\nhttps://phish2.example.com/verify\n"


class FakeResp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=65536):
        yield self.body


class FakeSession:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, url, timeout=None, stream=False, **kw):
        body, status = self.mapping.get(url, (b"", 404))
        return FakeResp(body, status)


def _cfg(tmp_path):
    return {
        "paths": {"raw_data_dir": str(tmp_path)},
        "sources": {"openphish_feed_url": "https://unit.invalid/feed.txt"},
        "network": {"feed_connect_timeout": 2, "feed_read_timeout": 5,
                    "feed_retries": 1, "feed_retry_backoff": 0,
                    "user_agent": "test", "max_feed_mb": 1},
    }


def test_snapshot_path_naming():
    assert snapshot_path(Path("/tmp"), dt.date(2026, 9, 6)) == \
        Path("/tmp/openphish_20260906.txt")


def test_accumulate_saves_today_snapshot(tmp_path):
    code, out = accumulate(_cfg(tmp_path),
                           session=FakeSession({"https://unit.invalid/feed.txt": (FEED, 200)}))
    assert code == 0 and out is not None and out.exists()
    assert out.name.startswith("openphish_") and out.name.endswith(".txt")
    assert out.read_bytes() == FEED


def test_accumulate_skips_existing_day_without_force(tmp_path):
    cfg = _cfg(tmp_path)
    today = dt.datetime.now(dt.timezone.utc).date()
    existing = snapshot_path(tmp_path, today)
    existing.write_text("PRESERVED", encoding="utf-8")
    code, out = accumulate(cfg, session=FakeSession({}))   # no feed mapping at all
    assert code == 0 and existing.read_text() == "PRESERVED"  # not refetched


def test_accumulate_force_overwrites(tmp_path):
    cfg = _cfg(tmp_path)
    today = dt.datetime.now(dt.timezone.utc).date()
    existing = snapshot_path(tmp_path, today)
    existing.write_text("OLD", encoding="utf-8")
    code, out = accumulate(cfg, force=True,
                           session=FakeSession({"https://unit.invalid/feed.txt": (FEED, 200)}))
    assert code == 0 and out.read_bytes() == FEED


def test_accumulate_failure_is_honest(tmp_path):
    code, out = accumulate(_cfg(tmp_path), session=FakeSession({}))  # feed 404
    assert code == 1 and out is None
    assert list(tmp_path.glob("openphish*.txt")) == []   # nothing written
