# 4D Kreuzer–Skarke: on-demand access and full-search feasibility

## Shipping today: v0.11 source-row direct access

- `/api/4d/live/row/{row_idx}` retrieves **one** record from the Hugging Face Dataset Server `/rows` endpoint using `offset=row_idx`, `length=1`.
- The frontend `/4d?row=12345` opens this record and displays the existing certified 4D primal/dual visualization.
- The browser has a source-row input and share/copy link action; it does not cache the dataset locally.
- **Do not confuse a row position with a canonical or permanent mathematical class ID.** Hugging Face could change source ordering or revisions. Existing URLs are therefore version-dependent.
- This direct access is bounded to `0..473800775`. Upstream limits, indexing, rate limiting, and availability still apply. On failure, the API returns an error; it never substitutes a different polytope.
- Filters in the current Hugging Face `/filter` provider still **might cover only the first 5 GB** of the total 15.8 GB dataset. Displaying a few hits is **not** proof of completeness.

## Next research step: remote Parquet query engine

The source contains separate Parquet shards by **vertex_count**, `polytopes-4d-05-vertices.parquet` (~27.5 KB) through much larger multi-GB shards. An exact vertex-count predicate can select which file to inspect. DuckDB's `httpfs` supports HTTP Range reads, projection of selected columns, and Parquet row-group predicate pushdown. However, arbitrary Hodge filters do not guarantee skipping most bytes; this depends on row-group statistics, column ordering, and selectivity.

**Feasibility must be measured, not assumed.** Start with the tiny shard and compare actual elapsed time/bytes/requests before proposing a full search service.

### Optional small-shard benchmark (developer workstation only)

Run outside production, e.g., in a disposable Python virtual environment on Windows:

```powershell
python -m venv .venv-4d
.\.venv-4d\Scripts\python.exe -m pip install --no-cache-dir duckdb
.\.venv-4d\Scripts\python.exe scripts\probe_4d_parquet.py --h11 101
```

The script queries **only the five-vertex file**, uses two DuckDB threads and a 256 MB memory limit, requests at most five rows, and doesn't download or store the whole dataset. DuckDB may install its small `httpfs` extension to a local extension directory; there is no promise of zero temporary files.

The probe does **not** measure transferred bytes (only wall-clock time). It proves basic connectivity/query semantics, not that scanning the 2 GB files will be cheap or fast.

### Before deploying a public complete-search backend

1. Capture actual HTTP Range request counts and bytes and verify `Accept-Ranges`/partial responses after Hugging Face/Xet redirection.
2. Benchmark at least the 05-, 10-, and 15-vertex shards **off the constrained webserver**, with resource/time budgets.
3. Verify that `WHERE` combinations over `vertex_count`, `facet_count`, `h11`, `h12` read a sufficiently small fraction of row groups; never infer completeness from HF viewer filters.
4. Use request rate limits, bounded concurrency, strict timeouts and cancellation; no unbounded SQL query or user-selected URL input.
5. Consider a separately hosted remote index if filtered remote scans remain expensive. Keep the webserver itself as a small API frontend.

## Sources

- Dataset and provenance: https://huggingface.co/datasets/calabi-yau-data/polytopes-4d
- Shard listing: https://huggingface.co/datasets/calabi-yau-data/polytopes-4d/tree/main
- Viewer limits: https://huggingface.co/docs/hub/en/data-studio
- Row endpoint: https://huggingface.co/docs/dataset-viewer/en/rows
- DuckDB Range I/O: https://duckdb.org/docs/stable/core_extensions/httpfs/https

## v0.12: bounded remote-Parquet search (experimental)

- The UI now uses `/api/4d/remote/search`, **not** the unreliable Dataset Viewer `/filter` endpoint.
- A request **must specify exactly one `vertex_count` in 5..15**. This is a **partial coverage pilot**; the remaining 4D shards (16+ vertices) have not yet been enabled. Never describe the pilot as exhaustive over 473,800,776 classes.
- Additional integer equality filters: `facet_count`, `point_count`, `dual_point_count`, `h11`, `h12`, `euler_characteristic`. All values are validated; server-side SQL identifiers and Parquet URLs are hardcoded/whitelisted.
- `/api/4d/remote/search?vertex_count=8&h11=101&limit=6` returns **up to 6 actual vertices records** plus `has_more`. There is no inferred global count or index. Page offsets max 10,000.
- `/api/4d/remote/count?vertex_count=15&h11=101` issues a separate **exact `COUNT(*)` over that shard only**. For 10+ vertices, an additional numeric filter is required to avoid accidental full scans. This count is only requested when the user clicks **Count all matches in this shard**.
- `/api/4d/remote/row/{vertex_count}/{shard_row}` fetches the physical file-relative row, if the provider can fulfill it within the timeout. Example UI URL: `/4d?shard=8&shard_row=42`.
- **Never treat `shard_row` as a global Hugging Face row index**. The existing v0.11 `/4d?row=...` links are kept, and continue using `/api/4d/live/row/{index}`.
- The Parquet source URL is fixed to `huggingface.co/datasets/calabi-yau-data/polytopes-4d`, specifically the `polytopes-4d-{05..15}-vertices.parquet` files. The source revision must be pinned before scientific reproducibility claims; the current `main` branch can change.
- One child process per request; 40-second subprocess timeout; one active query across multiple API workers via Linux file lock. Concurrent requests return 429; timeouts return 504; upstream failures return 503. Never silently return a partial row set as complete.
- DuckDB settings: two threads; 256 MB buffer manager; 32 MB temporary spill directory (removed after query); no persistent Parquet mirror or application-level results cache. **Neither network bytes nor total process RSS is hard-capped** by these settings. DuckDB installation and its `httpfs` extension add some disk usage to the Docker image/extension cache.
- Docker logs remain bounded by the `local` driver configuration from v0.10c.1. The existing persistent SQLite 3D volume is untouched.

### Deployment and smoke test

Apply v0.12 patch on the Windows development machine, commit, push, and run the existing `apps/polytope5d/deploy.ps1`. The patch appends `duckdb==2.0.1` to `api/requirements.txt` and registers the additional FastAPI router in `api/main.py`; it does not alter Compose or volumes.

```bash
cd /opt/polytope5d

docker compose exec -T api python -m unittest api.test_catalog4d_remote -v
# Optional real local Parquet test, run if DuckDB is installed by the Docker build:
docker compose exec -T api python -m unittest api.test_catalog4d_remote_db -v

curl -sS 'https://polytopes.mathematics.art/api/4d/remote/status'
curl -sS -i 'https://polytopes.mathematics.art/api/4d/remote/count?vertex_count=8&h11=101'
curl -sS -i 'https://polytopes.mathematics.art/api/4d/remote/search?vertex_count=8&h11=101&limit=2'
```

**Expected**: the 8-vertex count should be 2114 (as measured on the Windows development machine) if the hosted dataset has not changed. The measured 15-vertex count was 9828 (20.17 s; HTTPFS: 5.1 MiB; 75 GET). Only `COUNT(*)`, not actual vertex extraction, was benchmarked for large shards before this release. A real `/search` may be slower or return 504; do not infer completeness from such an error. The production integration and memory/transfer consumption must be observed before widening the allowed shard range.

To roll back: revert the Git commit and redeploy. Do **not** delete any volumes; no new database migration is involved.
