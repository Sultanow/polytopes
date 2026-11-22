"""Bounded 4D remote-Parquet experiment API (v0.12, opt-in, NOT replacing live API).

Never writes a local Parquet copy. Only a fixed set of Hugging Face shards is
reachable. DuckDB runs in a separate, timeout-killed process; a Linux flock
allows only ONE query in progress across API workers.

No request can claim that shard-local results exhaust the 473.8M-row catalogue.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import re
import json
import logging
import os
import subprocess
import sys
import tempfile

from fastapi import APIRouter, HTTPException, Path as ApiPath, Query

router = APIRouter(prefix="/api/4d/remote", tags=["4D bounded Parquet pilot"])
SOURCE = "calabi-yau-data/polytopes-4d"
# Optionally pin this dataset to a verified, immutable 40-character HF commit.
# Defaults to mutable 'main'; content fingerprint still protects deep links.
_REV = os.environ.get("POLYTOPES_4D_REVISION", "").strip().lower()
if _REV and not re.fullmatch(r"[0-9a-f]{40}", _REV):
    raise RuntimeError("POLYTOPES_4D_REVISION must be a 40-character lowercase commit SHA")
SOURCE_REVISION = _REV or "main"
REVISION_PINNED = bool(_REV)
BASE = f"https://huggingface.co/datasets/{SOURCE}/resolve/{SOURCE_REVISION}/"

# Phase 1: bounded, single-shard pilot; no implicit multi-shard scans.
SUPPORTED_SHARDS = tuple(range(5, 16))
FILTER_NAMES = (
    "facet_count", "point_count", "dual_point_count", "h11", "h12", "euler_characteristic"
)
RETURN_FIELDS = (
    "vertex_count", "facet_count", "point_count", "dual_point_count",
    "h11", "h12", "euler_characteristic"
)
WORKER_TIMEOUT_SECONDS = 40
LOCKFILE = "/tmp/mathematics-art-4d-search.lock"
MAX_OUTPUT_BYTES = 500_000  # includes <=12 records with <=128 vertices, enforced below


def row_fingerprint(row: dict) -> str:
    """Content fingerprint for detecting dataset/row changes; NOT a GL(4,Z) class hash."""
    keys = (*RETURN_FIELDS, "vertices")
    canonical = json.dumps({key: row[key] for key in keys},
                           sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def add_fingerprint(row: dict) -> dict:
    return {**row, "fingerprint": row_fingerprint(row)}


def parquet_url(vertex_count: int) -> str:
    if type(vertex_count) is not int or vertex_count not in SUPPORTED_SHARDS:
        raise ValueError("Choose a vertex count between 5 and 15")
    return f"{BASE}polytopes-4d-{vertex_count:02d}-vertices.parquet"


def filters_for_sql(vertex_count: int, values: dict) -> tuple[str, list[int]]:
    parquet_url(vertex_count)
    clauses = ["vertex_count = ?"]
    params = [vertex_count]
    for field in FILTER_NAMES:
        val = values.get(field)
        if val is None:
            continue
        if type(val) is not int or abs(val) > 1_000_000:
            raise ValueError("Invalid integer filter for " + field)
        clauses.append(f"{field} = ?")  # fixed, whitelist-only identifier
        params.append(val)
    return " AND ".join(clauses), params


def _duckdb_worker(payload: dict) -> dict:
    # DuckDB extension installation uses the *process* HOME, which may point
    # to a nonexistent /home/polytope in the slim API image.  Keep the tiny
    # httpfs extension cache in the container's ephemeral /tmp, not in /data.
    # This also works if this function is called directly by an offline test.
    duckdb_home = os.path.join(tempfile.gettempdir(), "mathematics-art-duckdb-home")
    os.makedirs(duckdb_home, mode=0o700, exist_ok=True)
    os.environ["HOME"] = duckdb_home

    # Not imported in API parent: only the short-lived child loads DuckDB.
    import duckdb

    vertex_count = payload["vertex_count"]
    mode = payload["mode"]
    url = parquet_url(vertex_count)
    # Direct row lookup has no filters in its payload, nor does it need any.
    # Only count and paginated search construct a filter predicate.
    if mode in ("count", "page"):
        clause, params = filters_for_sql(vertex_count, payload["filters"])
    with tempfile.TemporaryDirectory(prefix="math4d-") as tmp:
        con = duckdb.connect(":memory:")
        try:
            con.execute("SET threads=2")
            con.execute("SET memory_limit='256MB'")
            con.execute("SET max_temp_directory_size='32MB'")
            # Set both: home_directory for the DB instance and HOME for DuckDB's
            # extension installer (these locations are not interchangeable).
            con.execute("SET home_directory = '" + duckdb_home.replace("'", "''") + "'")
            # A local Parquet smoke test must not trigger httpfs auto-install.
            # Remote queries retain the HTTP timeout/retry restrictions.
            if url.startswith(("https://", "http://")):
                con.execute("SET http_timeout=12")
                con.execute("SET http_retries=0")
            con.execute("SET preserve_insertion_order=false")
            con.execute("SET enable_external_file_cache=false")
            # Prevent disk spill into the API's persistent SQLite volume.
            con.execute("SET temp_directory = '" + tmp.replace("'", "''") + "'")
            if mode == "count":
                sql = "SELECT COUNT(*) FROM read_parquet(?) WHERE " + clause
                exact = con.execute(sql, [url, *params]).fetchone()[0]
                return {"count": int(exact), "count_exact_for_shard": True}
            if mode == "page":
                # Keep physical, file-relative indexes so source links cannot be
                # confused with the Hugging Face global viewer row_idx.
                columns = ", ".join(("file_row_number AS shard_row", *RETURN_FIELDS, "vertices"))
                after_row = payload.get("after_row")
                if after_row is None:
                    # Legacy offset is kept so existing API consumers still work.
                    sql = (f"SELECT {columns} FROM read_parquet(?, file_row_number=true) "
                           f"WHERE {clause} ORDER BY shard_row LIMIT ? OFFSET ?")
                    args = [url, *params, payload["limit"] + 1, payload.get("offset", 0)]
                else:
                    sql = (f"SELECT {columns} FROM read_parquet(?, file_row_number=true) "
                           f"WHERE {clause} AND file_row_number > ? "
                           "ORDER BY shard_row LIMIT ?")
                    args = [url, *params, after_row, payload["limit"] + 1]
                page = con.execute(sql, args)
                rows = page.fetchall()
                result = []
                names = [d[0] for d in page.description]
                for values_row in rows[:payload["limit"]]:
                    item = dict(zip(names, values_row))
                    vertices = item.get("vertices")
                    if not (isinstance(vertices, list) and 5 <= len(vertices) <= 128
                            and all(isinstance(v, list) and len(v) == 4
                                    and all(type(x) is int and abs(x) < 10**9 for x in v)
                                    for v in vertices)):
                        raise ValueError("Unusable 4D vertices in provider row")
                    result.append(add_fingerprint(item))
                return {"items": result, "has_more": len(rows) > payload["limit"],
                        "next_cursor": result[-1]["shard_row"] if result and len(rows) > payload["limit"] else None,
                        "count_exact_for_shard": False}
            if mode == "row":
                sql = ("SELECT file_row_number AS shard_row, " + ", ".join((*RETURN_FIELDS, "vertices"))
                       + " FROM read_parquet(?, file_row_number=true) WHERE file_row_number = ? LIMIT 1")
                query = con.execute(sql, [url, payload["shard_row"]])
                row = query.fetchone()
                return {"item": add_fingerprint(dict(zip((d[0] for d in query.description), row))) if row else None}
            raise ValueError("Unsupported worker mode")
        finally:
            con.close()


def bounded_worker(payload: dict) -> dict:
    """Fail fast when busy; terminate worker on timeout; no worker in API process."""
    with open(LOCKFILE, "a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HTTPException(429, "One remote 4D query is already running; retry shortly") from exc
        command = [sys.executable, "-m", "api.catalog4d_remote", "--worker"]
        try:
            result = subprocess.run(
                command, input=json.dumps(payload), text=True, capture_output=True,
                timeout=WORKER_TIMEOUT_SECONDS, check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, "Remote Parquet query exceeded its 40-second limit. No complete result is claimed.") from exc
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    if result.returncode:
        logging.warning("4D remote worker failed: %s", result.stderr[-300:])
        # Do not reflect arbitrary provider logs to public callers.
        raise HTTPException(503, "Remote Parquet query failed; check API logs for the bounded worker error")
    if len(result.stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise HTTPException(502, "Result exceeded maximum API payload size")
    try:
        return json.loads(result.stdout)
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(502, "Invalid bounded-worker response") from exc


def validate_request(vertex_count, **values):
    try:
        filters_for_sql(vertex_count, values)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/status")
def remote_status():
    return {
        "engine": "duckdb-httpfs", "source": SOURCE,
        "source_revision": SOURCE_REVISION, "source_revision_pinned": REVISION_PINNED,
        "deep_links": "content-fingerprint checked; row position is not a mathematical class ID",
        "supported_vertex_counts": list(SUPPORTED_SHARDS),
        "single_shard_only": True,
        "dataset_complete": False,
        "count_exact_for_selected_shard_when_successful": True,
        "max_parallel_queries": 1, "timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "memory_limit": "256MB DuckDB buffer only (not a hard RSS limit)",
        "temporary_spill_limit": "32MB DuckDB temp; extension cache may use separate disk space",
        "network_byte_cap": False,
        "notice": "Pilot: no multi-shard query and no claim of exhaustive 473.8M-class coverage.",
    }


@router.get("/search")
def remote_search(
    vertex_count: int = Query(..., ge=5, le=15),
    offset: int = Query(0, ge=0, le=10_000),
    after_row: int | None = Query(None, ge=0, le=100_000_000),
    limit: int = Query(6, ge=1, le=12),
    facet_count: int | None = Query(None, ge=0, le=1_000_000),
    point_count: int | None = Query(None, ge=0, le=1_000_000),
    dual_point_count: int | None = Query(None, ge=0, le=1_000_000),
    h11: int | None = Query(None, ge=0, le=1_000_000),
    h12: int | None = Query(None, ge=0, le=1_000_000),
    euler_characteristic: int | None = Query(None, ge=-1_000_000, le=1_000_000),
):
    values = dict(facet_count=facet_count, point_count=point_count,
                  dual_point_count=dual_point_count, h11=h11, h12=h12,
                  euler_characteristic=euler_characteristic)
    # Direct Python calls in the offline unit tests retain FastAPI Query defaults.
    if not isinstance(after_row, int):
        after_row = None
    validate_request(vertex_count, **values)
    if after_row is not None and offset:
        raise HTTPException(422, "Use either after_row or offset, not both")
    if vertex_count >= 10 and all(x is None for x in values.values()):
        raise HTTPException(422, "For shards with 10+ vertices, specify at least one numeric filter")
    data = bounded_worker({"mode": "page", "vertex_count": vertex_count,
                           "filters": values, "offset": offset, "after_row": after_row,
                           "limit": limit})
    return {**data, "vertex_count": vertex_count, "offset": offset, "after_row": after_row,
            "limit": limit, "source": SOURCE, "source_revision": SOURCE_REVISION,
            "source_revision_pinned": REVISION_PINNED, "dataset_complete": False,
            "identity": "(vertex_count, shard_row) within source file/revision",
            "notice": "Exact predicates in this single Parquet shard; not an all-shard catalogue search."}


@router.get("/count")
def remote_count(
    vertex_count: int = Query(..., ge=5, le=15),
    facet_count: int | None = Query(None, ge=0, le=1_000_000),
    point_count: int | None = Query(None, ge=0, le=1_000_000),
    dual_point_count: int | None = Query(None, ge=0, le=1_000_000),
    h11: int | None = Query(None, ge=0, le=1_000_000),
    h12: int | None = Query(None, ge=0, le=1_000_000),
    euler_characteristic: int | None = Query(None, ge=-1_000_000, le=1_000_000),
):
    values = dict(facet_count=facet_count, point_count=point_count,
                  dual_point_count=dual_point_count, h11=h11, h12=h12,
                  euler_characteristic=euler_characteristic)
    validate_request(vertex_count, **values)
    if vertex_count >= 10 and all(x is None for x in values.values()):
        raise HTTPException(422, "For shards with 10+ vertices, specify at least one numeric filter")
    data = bounded_worker({"mode": "count", "vertex_count": vertex_count, "filters": values})
    return {**data, "source": SOURCE, "vertex_count": vertex_count,
            "source_revision": SOURCE_REVISION, "source_revision_pinned": REVISION_PINNED,
            "dataset_complete": False,
            "notice": "Exact count for selected shard only; not for the entire 4D catalogue."}


@router.get("/row/{vertex_count}/{shard_row}")
def remote_row(vertex_count: int = ApiPath(..., ge=5, le=15),
               shard_row: int = ApiPath(..., ge=0, le=100_000_000),
               fp: str | None = Query(None, min_length=64, max_length=64),
               rev: str | None = Query(None, min_length=40, max_length=40)):
    if not isinstance(fp, str):
        fp = None
    if not isinstance(rev, str):
        rev = None
    if rev is not None and (not re.fullmatch(r"[0-9a-f]{40}", rev) or rev != SOURCE_REVISION):
        raise HTTPException(409, "This link refers to another source revision; active dataset is different")
    if fp is not None and not re.fullmatch(r"[0-9a-f]{64}", fp):
        raise HTTPException(422, "Invalid row fingerprint")
    data = bounded_worker({"mode": "row", "vertex_count": vertex_count, "shard_row": shard_row})
    if data.get("item") is None:
        raise HTTPException(404, "Row does not exist in the selected Parquet shard")
    item = data["item"]
    if fp is not None and item["fingerprint"] != fp:
        raise HTTPException(409, "Source row changed. The stored link fingerprint no longer matches")
    return {**data, "source": SOURCE, "vertex_count": vertex_count,
            "source_revision": SOURCE_REVISION, "source_revision_pinned": REVISION_PINNED,
            "identity": "(vertex_count, shard_row)", "dataset_complete": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        parser.error("Internal worker only")
    try:
        payload = json.load(sys.stdin)
        result = _duckdb_worker(payload)
        output = json.dumps(result, separators=(",", ":"))
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise RuntimeError("Output exceeds limit")
        print(output)
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc)[:300], file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
