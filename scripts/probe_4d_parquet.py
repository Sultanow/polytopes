"""Optional, single-shard remote Parquet feasibility probe.

Run on a developer workstation with DuckDB installed (NOT in the production API).
It deliberately queries ONLY the tiny five-vertex shard. It is not a 473M search.
No Parquet data is written to local disk by this script.
"""
from __future__ import annotations

import argparse
import sys
import time

DATASET = "calabi-yau-data/polytopes-4d"
# The complete 4D dataset is far larger; the five-vertex shard is ~27.5 KB.
PARQUET = (
    "https://huggingface.co/datasets/calabi-yau-data/polytopes-4d/resolve/main/"
    "polytopes-4d-05-vertices.parquet"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only remote Parquet probe; five-vertex shard only")
    parser.add_argument("--h11", type=int, default=101, help="Hodge h11 equality filter (default: 101)")
    args = parser.parse_args()
    if not 0 <= args.h11 <= 1_000_000:
        parser.error("--h11 must be between 0 and 1000000")
    try:
        import duckdb
    except ImportError:
        print("DuckDB is not installed. See docs/4D-REMOTE-SEARCH.md", file=sys.stderr)
        return 2
    connection = duckdb.connect(":memory:")
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='256MB'")
    start = time.monotonic()
    try:
        # DuckDB httpfs uses HTTPS range requests; may install/load the httpfs extension.
        result = connection.execute(
            "SELECT vertex_count, facet_count, h11, h12, euler_characteristic "
            "FROM read_parquet(?) WHERE vertex_count = 5 AND h11 = ? LIMIT 5",
            [PARQUET, args.h11],
        ).fetchall()
    finally:
        connection.close()
    elapsed = time.monotonic() - start
    print("Source:", DATASET)
    print("Remote shard: polytopes-4d-05-vertices.parquet (~27.5 KB)")
    print("h11 =", args.h11)
    print("Rows (max 5):", result)
    print(f"Elapsed wall clock: {elapsed:.2f}s")
    print("Caution: this small-shard probe does not demonstrate fast/exhaustive scans of 2GB shards.")
    print("Network bytes and HTTP range request counts were NOT measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
