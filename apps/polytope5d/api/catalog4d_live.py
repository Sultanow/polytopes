"""4D Kreuzer–Skarke on-demand Hugging Face Dataset Viewer adapter.

No local dataset storage. Never claims comprehensive indexed searching.
Provider limitations (especially >5GB partial filter index) must be visible to users.
"""
from __future__ import annotations

import json
from urllib import error, parse, request

from fastapi import APIRouter, HTTPException, Path as ApiPath, Query

router = APIRouter(prefix="/api/4d/live", tags=["4D live source"])
SOURCE = "calabi-yau-data/polytopes-4d"
BASE = "https://datasets-server.huggingface.co"
# The dataset's webpage shows a single split called 'full'.
CONFIG = "default"
SPLIT = "full"
FIELDS = ("vertex_count", "facet_count", "point_count", "dual_point_count",
          "h11", "h12", "euler_characteristic")
MAX_RESPONSE_BYTES = 2_000_000
TOTAL_ROWS = 473_800_776


def predicates(**kwargs):
    """Only integer equality predicates for a fixed whitelist; never accept raw SQL."""
    result = []
    for key in FIELDS:
        value = kwargs.get(key)
        if value is None:
            continue
        if type(value) is not int or abs(value) > 1_000_000:
            raise ValueError(f"Invalid filter for {key}")
        result.append(f'"{key}"={value}')
    return " AND ".join(result)


def upstream_json(path: str, params: dict):
    """Bounded, fixed-host HTTPS GET; no arbitrary URL from the browser."""
    url = BASE + path + "?" + parse.urlencode(params)
    req = request.Request(url, headers={"Accept": "application/json", "User-Agent": "MathematicsArt-4D-live-pilot/0.10"})
    try:
        with request.urlopen(req, timeout=12) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise HTTPException(502, "Upstream response exceeds safety limit")
        return json.loads(body)
    except error.HTTPError as exc:
        # The external service may not have a full filter index for this large dataset.
        if exc.code in (400, 404, 422):
            raise HTTPException(503, "Hugging Face filter/rows endpoint unavailable for this dataset; try browsing or reduce filters") from exc
        if exc.code == 429:
            raise HTTPException(503, "Hugging Face is rate-limiting the live request") from exc
        raise HTTPException(502, f"Hugging Face returned HTTP {exc.code}") from exc
    except (error.URLError, TimeoutError, OSError) as exc:
        raise HTTPException(503, "Hugging Face is temporarily unreachable") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(502, "Hugging Face returned invalid JSON") from exc


def safe_row(item):
    """Return a small, whitelisted JSON record; ignore provider extras."""
    value = item.get("row") if isinstance(item, dict) else None
    if not isinstance(value, dict):
        return None
    vertices = value.get("vertices")
    if not (isinstance(vertices, list) and 5 <= len(vertices) <= 128 and
            all(isinstance(p, list) and len(p) == 4 and
                all(type(c) is int and abs(c) < 10**9 for c in p) for p in vertices)):
        return None
    idx = item.get("row_idx")
    if type(idx) is not int or idx < 0:
        return None
    result = {"row_idx": idx, "vertices": vertices}
    for key in FIELDS:
        v = value.get(key)
        result[key] = v if type(v) is int else None
    return result


@router.get("/status")
def status():
    """Inspect the viewer; do not misrepresent partial coverage as a full index."""
    upstream = upstream_json("/is-valid", {"dataset": SOURCE})
    return {
        "source": SOURCE, "provider": "huggingface-dataset-viewer",
        "viewer": bool(upstream.get("viewer")),
        "filter_supported_by_provider": bool(upstream.get("filter")),
        "search_supported_by_provider": bool(upstream.get("search")),
        "indexed_complete": False,
        "notice": "Provider filter/search index may cover only first 5 GB; results are not guaranteed complete across 473,800,776 rows.",
    }


@router.get("/search")
def search(
    offset: int = Query(0, ge=0, le=10_000),
    limit: int = Query(12, ge=1, le=24),
    vertex_count: int | None = Query(None, ge=5, le=100),
    facet_count: int | None = Query(None, ge=5, le=100),
    point_count: int | None = Query(None, ge=0, le=1_000_000),
    dual_point_count: int | None = Query(None, ge=0, le=1_000_000),
    h11: int | None = Query(None, ge=0, le=1_000_000),
    h12: int | None = Query(None, ge=0, le=1_000_000),
    euler_characteristic: int | None = Query(None, ge=-1_000_000, le=1_000_000),
):
    return search_page(offset=offset, limit=limit,
                       vertex_count=vertex_count, facet_count=facet_count,
                       point_count=point_count, dual_point_count=dual_point_count,
                       h11=h11, h12=h12, euler_characteristic=euler_characteristic)


def search_page(*, offset: int, limit: int, **filters):
    """Pure adapter entry point to simplify offline integration tests."""
    where = predicates(**filters)
    params = {"dataset": SOURCE, "config": CONFIG, "split": SPLIT,
              "offset": offset, "length": limit}
    if where:
        params["where"] = where
    upstream = upstream_json("/filter" if where else "/rows", params)
    original = upstream.get("rows")
    if not isinstance(original, list):
        raise HTTPException(502, "Unexpected rows format from Hugging Face")
    items = [row for item in original if (row := safe_row(item)) is not None]
    return {
        "items": items, "offset": offset, "limit": limit, "returned": len(items),
        "upstream_count": len(original), "filter": where or None,
        "source": SOURCE, "source_split": SPLIT, "full_dataset_rows": 473_800_776,
        "indexed_complete": False,
        "notice": "Filter results may be incomplete because the provider only indexes the first 5 GB of datasets this large.",
    }


@router.get("/row/{row_idx}")
def row_by_position(row_idx: int = ApiPath(..., ge=0, lt=TOTAL_ROWS)):
    """Fetch one arbitrary source position without walking or indexing the catalogue.

    A position is not a permanent class ID: the provider may change dataset order.
    No local caching or downloaded Parquet shards are needed.
    """
    params = {"dataset": SOURCE, "config": CONFIG, "split": SPLIT,
              "offset": row_idx, "length": 1}
    upstream = upstream_json("/rows", params)
    records = upstream.get("rows")
    if not isinstance(records, list):
        raise HTTPException(502, "Unexpected Hugging Face rows format")
    if len(records) != 1:
        raise HTTPException(404, "No row at this dataset position")
    selected = safe_row(records[0])
    if selected is None or selected["row_idx"] != row_idx:
        raise HTTPException(502, "Upstream row index or 4D coordinates are inconsistent")
    return {
        "item": selected, "source": SOURCE, "source_split": SPLIT,
        "source_row": row_idx, "full_dataset_rows": TOTAL_ROWS,
        "position_is_permanent_id": False,
        "notice": "This link refers to the current dataset row order, not a permanent mathematical class identifier.",
    }
