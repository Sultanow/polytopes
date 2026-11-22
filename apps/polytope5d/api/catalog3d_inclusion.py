"""Exact, independently checkable inclusion certificates for reflexive 3-polytopes.

The scan uses only the 48 signed coordinate permutations, a STRICT subset of
GL(3,Z). Every positive result is certified; missing results prove nothing.
The full 4319-class inclusion poset is NOT claimed by this implementation.
"""
from __future__ import annotations

import argparse
from itertools import permutations, product
import json
from pathlib import Path

from .db import connect, DB_PATH
from .catalog3d_store import ensure_schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS catalogue3d_inclusion (
    child_id INTEGER NOT NULL,
    parent_id INTEGER NOT NULL,
    witness_matrix TEXT NOT NULL,
    method TEXT NOT NULL,
    catalogue_sha256 TEXT NOT NULL,
    PRIMARY KEY (child_id, parent_id),
    FOREIGN KEY(child_id) REFERENCES catalogue3d(catalogue_id),
    FOREIGN KEY(parent_id) REFERENCES catalogue3d(catalogue_id),
    CHECK(child_id <> parent_id)
);
CREATE INDEX IF NOT EXISTS idx3di_parent ON catalogue3d_inclusion(parent_id, child_id);
"""


def det3(m):
    (a,b,c),(d,e,f),(g,h,i)=m
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)


def certificate_valid(child_vertices, parent_vertices, parent_normals, matrix, strict=True):
    """Check P -> U(P) subseteq Q exactly with integer arithmetic.

    Parent inequalities are <n,x> >= -1. All inputs must be integer matrices.
    Since all reflexive classes have unique interior lattice point 0, an affine
    lattice equivalence between reflexive polytopes has zero translation.
    """
    if (len(matrix)!=3 or any(len(row)!=3 for row in matrix)
            or any(type(x) is not int for row in matrix for x in row)
            or abs(det3(matrix))!=1):
        return False
    if (not child_vertices or not parent_vertices or not parent_normals
            or any(len(v)!=3 or any(type(x) is not int for x in v)
                   for v in child_vertices + parent_vertices + parent_normals)):
        return False
    images=[tuple(sum(matrix[r][c]*v[c] for c in range(3)) for r in range(3))
            for v in child_vertices]
    if any(sum(n[k]*v[k] for k in range(3)) < -1
           for v in images for n in parent_normals):
        return False
    return not strict or set(images) != {tuple(v) for v in parent_vertices}


def signed_permutation_matrices():
    """The 48 matrices in the signed permutation subgroup of GL(3,Z)."""
    for permutation in permutations(range(3)):
        for signs in product((-1,1),repeat=3):
            yield [[signs[r] if permutation[r]==c else 0 for c in range(3)]
                   for r in range(3)]


def ensure_inclusion_schema(db_path=DB_PATH):
    ensure_schema(db_path)
    with connect(db_path) as con:
        con.executescript(SCHEMA)


def load_records(con, start, end):
    rows=con.execute('SELECT catalogue_id,vertices,facet_normals FROM catalogue3d '
                     'WHERE catalogue_id >= ? AND catalogue_id < ? '
                     'ORDER BY catalogue_id',(start,end)).fetchall()
    return [{'id':row['catalogue_id'],
             'vertices':json.loads(row['vertices']),
             'normals':json.loads(row['facet_normals'])} for row in rows]


def bbox(vertices):
    return [(min(v[j] for v in vertices),max(v[j] for v in vertices))
            for j in range(3)]


def scan_ranges(source_start, source_end, target_start, target_end, db_path=DB_PATH):
    if not (0<=source_start<source_end<=4319 and 0<=target_start<target_end<=4319):
        raise ValueError('Ranges must be nonempty within catalogue IDs [0,4319)')
    ensure_inclusion_schema(db_path)
    with connect(db_path) as con:
        total=con.execute('SELECT COUNT(*) FROM catalogue3d').fetchone()[0]
        if total!=4319:
            raise ValueError(f'Requires the complete 4319-class import, found {total}')
        checksum=con.execute("SELECT value FROM catalogue3d_provenance WHERE name='source_file_sha256'").fetchone()
        if not checksum or not checksum[0]:
            raise ValueError('Cannot certify inclusions without a catalogue source SHA-256')
        catalogue_sha256=checksum[0]
        sources=load_records(con,source_start,source_end)
        targets=load_records(con,target_start,target_end)
        if len(sources)!=source_end-source_start or len(targets)!=target_end-target_start:
            raise ValueError('Missing catalogue IDs')
        # Per-target inequality and coordinate bounds. All are exact.
        target_bboxes={p['id']:bbox(p['vertices']) for p in targets}
        found=0
        tested=0
        for src in sources:
            seen=set()
            for mat in signed_permutation_matrices():
                images=[tuple(sum(mat[r][c]*v[c] for c in range(3)) for r in range(3))
                        for v in src['vertices']]
                img_minmax=bbox(images)
                for target in targets:
                    tid=target['id']
                    if tid == src['id'] or tid in seen:
                        continue
                    tested+=1
                    bounds=target_bboxes[tid]
                    if any(img_minmax[j][0]<bounds[j][0] or img_minmax[j][1]>bounds[j][1]
                           for j in range(3)):
                        continue
                    if not all(sum(n[k]*v[k] for k in range(3))>=-1
                               for n in target['normals'] for v in images):
                        continue
                    if set(images)=={tuple(v) for v in target['vertices']}:
                        continue
                    # Recheck via independent public certificate predicate.
                    if not certificate_valid(src['vertices'],target['vertices'],target['normals'],mat):
                        raise AssertionError('Internal witness verification failed')
                    con.execute('INSERT OR REPLACE INTO catalogue3d_inclusion '
                                '(child_id,parent_id,witness_matrix,method,catalogue_sha256) '
                                'VALUES(?,?,?,?,?)',
                                (src['id'],tid,json.dumps(mat,separators=(',',':')),
                                 'signed_coordinate_permutation',catalogue_sha256))
                    seen.add(tid)
                    found+=1
        con.commit()
    return {'scanned_sources':len(sources),'scanned_targets':len(targets),
            'certified_relations_in_range':found,
            'candidate_checks':tested,
            'method':'signed_coordinate_permutations_only',
            'complete_poset':False}


def main():
    parser=argparse.ArgumentParser(description='Discover a PARTIAL set of exact 3D inclusions')
    parser.add_argument('--source-start',type=int,default=0)
    parser.add_argument('--source-end',type=int,default=40)
    parser.add_argument('--target-start',type=int,default=0)
    parser.add_argument('--target-end',type=int,default=150)
    parser.add_argument('--db',type=Path,default=DB_PATH)
    args=parser.parse_args()
    print(json.dumps(scan_ranges(args.source_start,args.source_end,
                                 args.target_start,args.target_end,args.db),indent=2))


if __name__=='__main__':
    main()
