"""CLI-only ingestion of a *converted* research-catalogue subset in JSONL format.

Call only from the server shell; there is deliberately no HTTP write endpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .db import COLUMNS, connect, init_db, normalize_record, serialize


def import_file(path: Path, db_path: Path | None = None) -> int:
    if db_path is None:
        from .db import DB_PATH
        db_path=DB_PATH
    init_db(db_path)
    count=0
    with connect(db_path) as con, path.open("rt",encoding="utf-8") as f:
        sql="INSERT OR REPLACE INTO polytopes ("+",".join(COLUMNS)+") VALUES ("+",".join("?" for _ in COLUMNS)+")"
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            if len(line)>2_000_000:
                raise ValueError(f"line {line_no}: record too large")
            try:
                record=normalize_record(json.loads(line))
            except (ValueError,TypeError,KeyError) as exc:
                raise ValueError(f"line {line_no}: {exc}") from exc
            con.execute(sql,serialize(record))
            count+=1
            if count%1000==0:con.commit()
    return count

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Import a validated JSONL subset into the local staging index")
    parser.add_argument("path",type=Path)
    args=parser.parse_args()
    print(f"Imported {import_file(args.path)} rows")
