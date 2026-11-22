"""One-time PALP/SageMath 3D normal-form import into SQLite.

No dataset is bundled or mirrored by this source repository. After installing
`sagemath-data-polytopes==20170220.1`, run in API container:
    python -m api.catalog3d_import

Alternatively supply --file /path/to/reflexive_polytopes_3d with PALP matrices.

The 3D database's redistribution conditions need clarification from its authors.
"""
from __future__ import annotations
import argparse
import hashlib
import re
import sys
from io import StringIO

from .catalog3d_geometry import exact_geometry
from .catalog3d_store import SCHEMA, record_to_row
from .db import DB_PATH, connect

HEADER = re.compile(r'^\s*(\d+)\s+(\d+)(?:\s+.*)?$')


def read_palp_matrices(text):
    """Parse SageMath's line-oriented PALP vertex-matrix stream.

    Each block: `3 N [comment]` followed by three rows of N signed integers;
    columns of the matrix are the lattice vertices. Error on malformed input.
    """
    lines = iter(enumerate(StringIO(text),1))
    for line_no, line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match=HEADER.match(line)
        if not match:
            raise ValueError(f'PALP header at line {line_no}: {line[:120]}')
        dimension, n = map(int, match.groups())
        if dimension != 3 or not 4 <= n <= 128:
            raise ValueError(f'Unexpected PALP matrix shape {dimension}x{n} at line {line_no}')
        rows=[]
        for r in range(3):
            try: row_no, row = next(lines)
            except StopIteration as e: raise ValueError(f'Truncated matrix after line {line_no}') from e
            parts=row.split()
            if len(parts) != n:
                raise ValueError(f'PALP line {row_no}: expected {n} entries, got {len(parts)}')
            try: rows.append([int(x) for x in parts])
            except ValueError as e: raise ValueError(f'Noninteger vertex at PALP line {row_no}') from e
        yield [list(p) for p in zip(*rows)]


def import_text(text, db_path=DB_PATH, expected_count=4319):
    """All-or-nothing import. Count and every primal/dual graph verified."""
    checksum=hashlib.sha256(text.encode('utf-8')).hexdigest()
    with connect(db_path) as con:
        con.executescript(SCHEMA)
        con.execute('DELETE FROM catalogue3d')
        con.execute('DELETE FROM catalogue3d_provenance')
        count=0
        for count, vertices in enumerate(read_palp_matrices(text),1):
            if count > 4319:
                raise ValueError('More than 4319 records: wrong input file')
            try:
                geometry=exact_geometry(vertices)
            except ValueError as e:
                raise ValueError(f'Invalid 3D lattice polytope at Sage index {count-1}: {e}') from e
            con.execute('INSERT INTO catalogue3d VALUES (?,?,?,?,?,?,?,?,?,?)',
                        record_to_row(count-1,geometry))
        if expected_count is not None and count!=expected_count:
            raise ValueError(f'Expected {expected_count} classes; imported {count}. Transaction rolled back.')
        for key,value in [('source','SageMath polytopes_db 20170220 / Kreuzer-Skarke'),
                          ('source_file_sha256',checksum),('count',str(count)),
                          ('indexing','Sage ReflexivePolytope(3, i), zero-based')]:
            con.execute('INSERT INTO catalogue3d_provenance(name,value) VALUES(?,?)',(key,value))
    return {'count':count,'sha256':checksum}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Import all 4319 SageMath reflexive 3-polytopes')
    parser.add_argument('--file',help='PALP matrix-file override; uses sagemath-data-polytopes if omitted')
    parser.add_argument('--db',default=str(DB_PATH))
    parser.add_argument('--allow-partial',action='store_true',help='TESTS ONLY: do not claim full catalogue')
    args=parser.parse_args(argv)
    if args.file:
        from pathlib import Path
        text=Path(args.file).read_text(encoding='utf-8')
    else:
        try:
            from sage_data_polytopes import open_reflexive_polytopes
        except ImportError as e:
            raise SystemExit('Install sagemath-data-polytopes==20170220.1 first') from e
        with open_reflexive_polytopes(3) as stream:
            text=stream.read()
        if isinstance(text, bytes):
            text=text.decode('utf-8')
    print('Reading and exactly checking 3D PALP matrices...',flush=True)
    result=import_text(text,args.db,None if args.allow_partial else 4319)
    print(f'OK: {result["count"]} validated 3D classes, sha256={result["sha256"]}',flush=True)
    if args.allow_partial: print('TEST DATA ONLY: do not advertise a complete classification.',flush=True)

if __name__ == '__main__':
    main()
