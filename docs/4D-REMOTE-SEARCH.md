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
