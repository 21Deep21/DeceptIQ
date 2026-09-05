"""SQLite scan history (data/scans.db).

Every analyzed URL (single or batch) is recorded once per scan with:
timestamp, sanitized URL, verdict, calibrated probability, severity,
classification threshold, model version, SHAP backend, top SHAP
features (JSON) and the IOC summary (JSON). Credentials are redacted
BEFORE storage (features.ioc_extraction sanitisation) - the database
never contains raw passwords.

Connections are opened per operation: low request volume, and it
avoids sqlite3's default thread affinity under the threaded dev
server. WAL mode is enabled at initialisation.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("app.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at TEXT NOT NULL,
    url TEXT NOT NULL,
    verdict TEXT NOT NULL,
    probability REAL NOT NULL,
    severity TEXT NOT NULL,
    threshold REAL NOT NULL,
    model_version TEXT,
    shap_backend TEXT,
    top_shap_json TEXT,
    ioc_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_time ON scans(scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_verdict ON scans(verdict);
CREATE INDEX IF NOT EXISTS idx_scans_severity ON scans(severity);
"""


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _escape_like(q: str) -> str:
    return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _safe_json(text: Optional[str]) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class ScanStore:
    def __init__(self, path: str):
        self._path = str(path)
        pathlib.Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()
        logger.info("scan store ready: %s", self._path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, *, url: str, verdict: str, probability: float, severity: str,
               threshold: float, model_version: str, shap_backend: str,
               top_shap: Dict[str, Any], ioc: Dict[str, Any]) -> Tuple[int, str]:
        scanned_at = utc_now_iso()
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO scans (scanned_at, url, verdict, probability, severity,"
                " threshold, model_version, shap_backend, top_shap_json, ioc_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (scanned_at, url, verdict, float(probability), severity,
                 float(threshold), model_version, shap_backend,
                 json.dumps(top_shap, default=str),
                 json.dumps(ioc, default=str)),
            )
            conn.commit()
            scan_id = int(cur.lastrowid)
        finally:
            conn.close()
        return scan_id, scanned_at

    def list(self, limit: int = 50, verdict: Optional[str] = None,
             severity: Optional[str] = None, q: Optional[str] = None,
             order: str = "desc") -> Tuple[List[Dict[str, Any]], int]:
        where: List[str] = []
        params: List[Any] = []
        if verdict is not None:
            where.append("verdict = ?")
            params.append(verdict)
        if severity is not None:
            where.append("severity = ?")
            params.append(severity)
        if q:
            where.append("url LIKE ? ESCAPE '\\'")
            params.append("%" + _escape_like(q) + "%")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        order_sql = "DESC" if order == "desc" else "ASC"
        conn = self._connect()
        try:
            total = int(conn.execute(
                f"SELECT COUNT(*) FROM scans{where_sql}", params).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM scans{where_sql} "
                f"ORDER BY scanned_at {order_sql}, id {order_sql} LIMIT ?",
                params + [max(1, int(limit))],
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_dict(r) for r in rows], total

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "scan_id": int(row["id"]),
            "scanned_at": row["scanned_at"],
            "url": row["url"],
            "verdict": row["verdict"],
            "probability": row["probability"],
            "severity": row["severity"],
            "threshold": row["threshold"],
            "model_version": row["model_version"],
            "shap_backend": row["shap_backend"],
            "top_shap": _safe_json(row["top_shap_json"]),
            "ioc": _safe_json(row["ioc_json"]),
        }
