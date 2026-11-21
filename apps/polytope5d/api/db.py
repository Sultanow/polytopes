"""Small, read-heavy SQLite index. A *staging* index, not the billion-row catalogue."""
from __future__ import annotations
import itertools
from contextlib import contextmanager
import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("POLYTOPE_DB", "/data/polytopes.sqlite3"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS polytopes (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  dimension INTEGER NOT NULL DEFAULT 5 CHECK (dimension=5),
  reflexive INTEGER NOT NULL CHECK (reflexive IN (0,1)),
  weights TEXT,
  weight_key TEXT,
  vertex_count INTEGER,
  facet_count INTEGER,
  point_count INTEGER,
  frequency INTEGER,
  vertices_5d TEXT,
  edges TEXT,
  facet_normals TEXT,
  info TEXT
);
CREATE INDEX IF NOT EXISTS idx_polytope_counts ON polytopes (reflexive,vertex_count,facet_count,id);
CREATE INDEX IF NOT EXISTS idx_polytope_weights ON polytopes (weight_key);
CREATE INDEX IF NOT EXISTS idx_polytope_source ON polytopes (source,id);
"""

COLUMNS = ("id", "source", "dimension", "reflexive", "weights", "weight_key",
           "vertex_count", "facet_count", "point_count", "frequency",
           "vertices_5d", "edges", "facet_normals", "info")
JSON_COLUMNS = ("weights", "vertices_5d", "edges", "facet_normals", "info")


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    con = sqlite3.connect(str(db_path), timeout=15)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=15000")
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(db_path: Path | str = DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as con:
        con.executescript(SCHEMA)
        # Exactly constructed demonstrators. Do not count as catalogue records.
        if con.execute("SELECT 1 FROM polytopes WHERE source='analytic_demo' LIMIT 1").fetchone() is None:
            con.executemany(
                "INSERT INTO polytopes (" + ",".join(COLUMNS) + ") VALUES (" + ",".join("?" for _ in COLUMNS) + ")",
                [serialize(r) for r in demo_records()],
            )


def serialize(obj: dict) -> tuple:
    return tuple(json.dumps(obj.get(k), separators=(",", ":")) if k in JSON_COLUMNS and obj.get(k) is not None
                 else obj.get(k) for k in COLUMNS)


def deserialize(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    for name in JSON_COLUMNS:
        if data[name] is not None:
            data[name] = json.loads(data[name])
    data["reflexive"] = bool(data["reflexive"])
    data.pop("weight_key", None)
    return data


def simplex(q: tuple[int,...]) -> dict:
    s = 1 + sum(q)
    assert len(q) == 5 and all(s % x == 0 for x in q)
    vertices = [[-x for x in q]] + [[int(i == j) for j in range(5)] for i in range(5)]
    normals = [[-1] * 5] + [[s // q[i] - 1 if i == j else -1 for j in range(5)] for i in range(5)]
    weights = [1, *q]
    return {
      "id": "demo-simplex-" + "-".join(map(str,q)), "source": "analytic_demo", "dimension": 5,
      "reflexive": 1, "weights": weights, "weight_key": ",".join(map(str,weights)),
      "vertex_count": 6, "facet_count": 6, "point_count": None, "frequency": None,
      "vertices_5d": vertices, "edges": [list(pair) for pair in itertools.combinations(range(6),2)],
      "facet_normals": normals, "info": {"construction": "conv(-q,e1,...,e5)", "S": s,
      "proof": "every q_i divides S; exact integral polar normals"}
    }


def demo_records():
    for q in itertools.combinations_with_replacement(range(1,13), 5):
        if all((1+sum(q)) % x == 0 for x in q):
            yield simplex(q)
    # Hypercube [-1,1]^5. Facets: ±x_i<=1; polar is the integral cross-polytope.
    cube = list(itertools.product((-1,1), repeat=5))
    cube_edges = [[i,j] for i,v in enumerate(cube) for j,w in enumerate(cube)
                  if i<j and sum(a!=b for a,b in zip(v,w))==1]
    yield {"id":"demo-hypercube-5d", "source":"analytic_demo", "dimension":5,
       "reflexive":1, "weights":None, "weight_key":None, "vertex_count":32, "facet_count":10,
       "point_count":243, "frequency":None, "vertices_5d":[list(v) for v in cube], "edges":cube_edges,
       "facet_normals":None,"info":{"construction":"[-1,1]^5", "polar":"5D cross-polytope"}}
    cross = [[s*int(i==j) for j in range(5)] for i in range(5) for s in (-1,1)]
    cross_edges = [[i,j] for i in range(10) for j in range(i+1,10) if i//2!=j//2]
    yield {"id":"demo-crosspolytope-5d", "source":"analytic_demo", "dimension":5,
       "reflexive":1, "weights":None, "weight_key":None, "vertex_count":10, "facet_count":32,
       "point_count":11, "frequency":None, "vertices_5d":cross, "edges":cross_edges,
       "facet_normals":None,"info":{"construction":"conv(±e1,...,±e5)", "polar":"5D cube"}}


def normalize_record(obj: dict) -> dict:
    """Validate the *adapter JSONL*, NOT the source catalogue's unpublished format."""
    if not isinstance(obj, dict) or not isinstance(obj.get("id"),str) or not 1<=len(obj["id"])<=128:
        raise ValueError("Each record requires a nonempty id (max 128 chars)")
    if obj["id"].startswith("demo-"):
        raise ValueError("Imported IDs may not start with demo-")
    source = obj.get("source")
    if not isinstance(source,str) or not 1<=len(source)<=100 or source=="analytic_demo":
        raise ValueError("source must identify an external dataset, not analytic_demo")
    if obj.get("dimension",5)!=5 or not isinstance(obj.get("reflexive"), bool):
        raise ValueError("dimension must be 5; reflexive must be a boolean")
    weights=obj.get("weights")
    if weights is not None and (not isinstance(weights,list) or len(weights)!=6 or any(type(w)!=int or w<1 for w in weights)):
        raise ValueError("weights must be six positive integers or null")
    for key in ("vertex_count", "facet_count", "point_count", "frequency"):
        val=obj.get(key)
        if val is not None and (type(val)!=int or val<0 or val>2**63-1):
            raise ValueError(f"{key} must be a nonnegative int64 or null")
    vertices=obj.get("vertices_5d")
    if vertices is not None:
        if not isinstance(vertices,list) or len(vertices)>2048 or any(
                not isinstance(v,list) or len(v)!=5 or any(type(x)!=int for x in v) for v in vertices):
            raise ValueError("vertices_5d must be <=2048 integer 5-vectors")
        if obj.get("vertex_count") is not None and obj["vertex_count"]!=len(vertices):
            raise ValueError("vertex_count does not match supplied vertices")
    edges=obj.get("edges")
    if edges is not None and (vertices is None or not isinstance(edges,list) or len(edges)>100_000 or any(
            not isinstance(e,list) or len(e)!=2 or any(type(i)!=int or i<0 or i>=len(vertices) for i in e) or e[0]==e[1]
            for e in edges)):
        raise ValueError("edges need integer indices into vertices_5d")
    normals=obj.get("facet_normals")
    if normals is not None and (not isinstance(normals,list) or len(normals)>2048 or any(
            not isinstance(v,list) or len(v)!=5 or any(type(x)!=int for x in v) for v in normals)):
        raise ValueError("facet_normals must be <=2048 integer 5-vectors")
    info=obj.get("info")
    if info is not None and (not isinstance(info,dict) or len(json.dumps(info))>16_384):
        raise ValueError("info must be a small JSON object")
    return {"id":obj["id"],"source":source,"dimension":5,"reflexive":int(obj["reflexive"]),
      "weights":weights,"weight_key":",".join(map(str, weights)) if weights else None,
      "vertex_count":obj.get("vertex_count") or (len(vertices) if vertices is not None else None),
      "facet_count":obj.get("facet_count"),"point_count":obj.get("point_count"),
      "frequency":obj.get("frequency"),"vertices_5d":vertices,"edges":edges,
      "facet_normals":normals,"info":info}
