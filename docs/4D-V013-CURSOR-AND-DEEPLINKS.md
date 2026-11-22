# 4D v0.13 — cursor paging and verifiable source-row links

This update builds on the working v0.12.2 DuckDB remote-Parquet pilot. It does **not** import a 4D dataset locally.

## Features and guarantees

- **Seek pagination:** `GET /api/4d/remote/search?vertex_count=8&h11=101&limit=6&after_row=1234` returns rows strictly after physical `file_row_number=1234`, plus `next_cursor` if another page exists. The first page omits `after_row`. Previous pages are navigated in the React UI using a stack of observed cursors. This removes the old 10,000-result offset ceiling **in the UI**, but does not guarantee a sub-second query for every page. The legacy `offset` API continues to work with its previous cap for backwards compatibility. Never provide both parameters in one request.
- **Tamper/change-detecting links:** Each item has `fingerprint`, a SHA-256 digest over its original `vertices` and seven numeric metadata fields (canonical sorted JSON). Links look like `/4d?shard=8&shard_row=1234&fp=<sha256>`. On page reload the API requires the fingerprint to match. A changed row returns **HTTP 409** instead of silently displaying a different polytope. This is **not** a `GL(4,Z)`-invariant mathematical class identifier; coordinate changes can change the fingerprint even for equivalent polytopes.
- **Source revision provenance:** By default URLs still use `resolve/main/`, which is **mutable**. The status API explicitly reports `source_revision:"main"`, `source_revision_pinned:false`. Operators can optionally configure `POLYTOPES_4D_REVISION=<verified 40-hex-digit Hugging Face dataset commit SHA>` in the API container environment, which makes all DuckDB Parquet reads use `resolve/<sha>/` and marks the revision pinned. When pinned, new shared links add `rev=<sha>` and fail with 409 if opened on a deployment configured with a different revision. **Do not use the Dataset Viewer `x-revision` header as proof of repository SHA; check the dataset repository API or commit history.** A pinned revision must actually exist and have all queried shard filenames.
- **Same limits:** Single-shard only, vertex counts 5–15, at most one running query, 40-second process timeout, DuckDB's 256 MB buffer limit, 32 MB temp spill limit; total RSS and network bytes remain **not hard-capped**. Count queries still return exact totals only within the selected shard, not the entire 473.8-million-class dataset. No new permanent storage or extra container.
- **Compatibility:** Existing `/4d?row=N` links still use the Hugging Face dataset viewer. Existing `/4d?shard=V&shard_row=N` links remain valid but, without `fp`, are not change-detecting. Existing v0.12 `/api/4d/remote/count`, `/search` (offset), and `/row` endpoints still work.

## End-to-end server smoke tests

```bash
cd /opt/polytope5d

docker compose exec -T api python -m unittest api.test_catalog4d_remote api.test_catalog4d_v013 -v
docker compose exec -T api python -m unittest api.test_catalog4d_remote_db -v

# First full page; inspect next_cursor and fingerprint
curl -fsS --max-time 50 \
 'https://polytopes.mathematics.art/api/4d/remote/search?vertex_count=8&h11=101&limit=2'

# Replace cursor with the next_cursor from previous JSON response
curl -fsS --max-time 50 \
 'https://polytopes.mathematics.art/api/4d/remote/search?vertex_count=8&h11=101&limit=2&after_row=123'

curl -fsS --max-time 50 \
 'https://polytopes.mathematics.art/api/4d/remote/status'
```

Expected count remains **2114** for `vertex_count=8&h11=101` in the previously tested dataset revision, provided upstream data has not changed. The **real remote page with coordinates** must still be verified after deployment; previous v0.12.2 validation established count and direct-row, not the actual page response.

## Validation, rollback

The PowerShell script checks the exact patch against the current repository without modifying unrelated landing-page files. Run `apply-v013.ps1 -CheckOnly`, then apply, commit, push and use the existing `deploy.ps1`. Rolling back the v0.13 commit and redeploying restores v0.12.2. Do not remove Docker volumes; the 3D SQLite catalogue is unrelated.
