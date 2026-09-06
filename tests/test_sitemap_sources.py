import gzip

import requests

from dataset.sitemap_sources import (
    collect_sitemap_urls,
    maybe_gunzip,
    parse_sitemap,
    robots_sitemap_lines,
)

URLSET = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    b"<url><loc>https://example.org/a</loc></url>"
    b"<url><loc>https://example.org/docs/guide?x=1</loc></url>"
    b"<url><loc>https://example.org/b</loc></url>"
    b"</urlset>"
)
INDEX = (
    b'<?xml version="1.0"?>'
    b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    b"<sitemap><loc>https://example.org/sitemap-1.xml</loc></sitemap>"
    b"<sitemap><loc>https://example.org/sitemap-2.xml</loc></sitemap>"
    b"</sitemapindex>"
)


def test_robots_sitemap_lines():
    robots = (
        "User-agent: *\n"
        "Disallow: /private\n"
        "Sitemap: https://example.org/sitemap.xml\n"
        "sitemap: https://example.org/news.xml\n"
    )
    assert robots_sitemap_lines(robots) == [
        "https://example.org/sitemap.xml", "https://example.org/news.xml"]
    assert robots_sitemap_lines("User-agent: *\nDisallow: /\n") == []
    assert robots_sitemap_lines("") == []


def test_maybe_gunzip_passthrough_and_roundtrip():
    assert maybe_gunzip(b"<xml/>") == b"<xml/>"
    gz = gzip.compress(URLSET)
    assert maybe_gunzip(gz) == URLSET


def test_parse_sitemap_urlset_pages():
    pages, children, is_index, err = parse_sitemap(URLSET)
    assert err is None and not is_index and children == []
    assert pages == [
        "https://example.org/a", "https://example.org/docs/guide?x=1",
        "https://example.org/b",
    ]


def test_parse_sitemap_index_children():
    pages, children, is_index, err = parse_sitemap(INDEX)
    assert err is None and is_index and pages == []
    assert children == [
        "https://example.org/sitemap-1.xml", "https://example.org/sitemap-2.xml"]


def test_parse_sitemap_without_namespace():
    plain = b"<urlset><url><loc>https://example.org/x</loc></url></urlset>"
    pages, _, is_index, err = parse_sitemap(plain)
    assert err is None and not is_index and pages == ["https://example.org/x"]


def test_parse_sitemap_respects_max_locs():
    big = b"<urlset>" + b"".join(
        f"<url><loc>https://example.org/p{i}</loc></url>".encode() for i in range(10)
    ) + b"</urlset>"
    pages, _, _, err = parse_sitemap(big, max_locs=4)
    assert err is None and len(pages) == 4


def test_parse_sitemap_gzipped_file():
    pages, _, is_index, err = parse_sitemap(gzip.compress(URLSET))
    assert err is None and not is_index and len(pages) == 3


def test_parse_sitemap_malformed_xml_returns_error_note():
    pages, children, is_index, err = parse_sitemap(b"<urlset><url>")
    assert err is not None  # reported, never raised


class FakeResp:
    def __init__(self, body, status=200):
        self.body = body
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for url")

    def iter_content(self, chunk_size=65536):
        yield self.body


class FakeSession:
    """Test double for requests.Session.

    CONTRACT: every mapping value MUST be a (body, status) 2-tuple.
    (A previous version of this fixture used 1-tuples and raw bytes,
    which raised ValueError inside FakeSession.get - silently caught
    by _get_bytes - making every collect test return []. The pure
    parse tests never touch this class, which is why they passed.)
    """

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def get(self, url, timeout=None, stream=False, **kw):
        self.calls.append(url)
        body, status = self.mapping.get(url, (b"", 404))
        return FakeResp(body, status)


def _cfg(**overrides):
    sm = {
        "request_delay_seconds": 0,
        "max_file_mb": 1,
        "max_locs_per_file": 100,
        "max_urls_per_domain": 200,
        "max_sitemap_files_per_domain": 3,
        "max_children_per_index": 15,
        "fallback_sitemap_paths": ["sitemap.xml"],
        "domains": ["example.org"],
    }
    sm.update(overrides)
    return {"sitemaps": sm, "network": {"feed_connect_timeout": 2, "feed_read_timeout": 5}}


def test_collect_via_robots_declared_sitemap():
    session = FakeSession({
        "https://example.org/robots.txt": (
            b"User-agent: *\nSitemap: https://example.org/sitemap.xml\n", 200),
        "https://example.org/sitemap.xml": (URLSET, 200),
    })
    entries, report = collect_sitemap_urls(_cfg(), session)
    assert [e["url"] for e in entries] == [
        "https://example.org/a", "https://example.org/docs/guide?x=1",
        "https://example.org/b"]
    assert all(e["label"] == 0 and e["source"] == "sitemap" for e in entries)
    assert report["domains_ok"] == 1 and report["domains_failed"] == 0


def test_collect_via_fallback_when_robots_missing():
    session = FakeSession({
        "https://example.org/sitemap.xml": (gzip.compress(URLSET), 200),
    })
    entries, report = collect_sitemap_urls(_cfg(), session)
    assert len(entries) == 3  # robots.txt 404 -> fallback path worked
    assert report["domains_ok"] == 1


def test_collect_respects_per_domain_cap_and_scheme_filter():
    urlset = b"<urlset>" + b"".join(
        f"<url><loc>https://example.org/page{i}</loc></url>".encode() for i in range(10)
    ) + b"<url><loc>ftp://example.org/skip-me</loc></url></urlset>"
    session = FakeSession({
        "https://example.org/robots.txt": (
            b"Sitemap: https://example.org/sitemap.xml\n", 200),
        "https://example.org/sitemap.xml": (urlset, 200),
    })
    entries, _ = collect_sitemap_urls(_cfg(max_urls_per_domain=5), session)
    urls = [e["url"] for e in entries]
    assert len(urls) == 5
    assert all(u.startswith("https://") for u in urls)  # non-http(s) filtered
