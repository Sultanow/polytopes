"""Dedicated 3D index inside the EXISTING SQLite volume; 5D tables are untouched."""
from __future__ import annotations
import json
from pathlib import Path
from .db import connect, DB_PATH

SCHEMA = '''
CREATE TABLE IF NOT EXISTS catalogue3d (
  catalogue_id INTEGER PRIMARY KEY CHECK (catalogue_id BETWEEN 0 AND 4318),
  vertex_count INTEGER NOT NULL,
  edge_count INTEGER NOT NULL,
  facet_count INTEGER NOT NULL,
  vertices TEXT NOT NULL,
  edges TEXT NOT NULL,
  facets TEXT NOT NULL,
  facet_normals TEXT NOT NULL,
  dual_edges TEXT NOT NULL,
  dual_facets TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx3d_counts ON catalogue3d (vertex_count,facet_count,catalogue_id);
CREATE INDEX IF NOT EXISTS idx3d_facets ON catalogue3d (facet_count,vertex_count,catalogue_id);
CREATE TABLE IF NOT EXISTS catalogue3d_provenance (
  name TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
'''


def ensure_schema(db_path: Path | str=DB_PATH):
    with connect(db_path) as con:
        con.executescript(SCHEMA)


def record_to_row(index, geometry):
    return (index, geometry['vertex_count'], geometry['edge_count'], geometry['facet_count'],
            *(json.dumps(geometry[key],separators=(',',':')) for key in
              ('vertices','edges','facets','facet_normals','dual_edges','dual_facets')))


def expand(row):
    obj = dict(row)
    for key in ('vertices','edges','facets','facet_normals','dual_edges','dual_facets'):
        obj[key] = json.loads(obj[key])
    obj['id'] = f'ks3d-{obj["catalogue_id"]:04d}'
    obj['source'] = 'SageMath / Kreuzer-Skarke 3D'
    obj['dimension'] = 3
    obj['reflexive'] = True
    obj['dual_vertices'] = obj['facet_normals']
    obj['dual_vertex_count'] = obj['facet_count']
    obj['dual_edge_count'] = len(obj['dual_edges'])
    obj['dual_facet_count'] = obj['vertex_count']
    return obj
