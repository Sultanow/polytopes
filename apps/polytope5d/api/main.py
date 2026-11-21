"""Public, read-only catalogue search API. Bounded inputs, no import endpoint."""
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from .db import connect, deserialize, init_db
from .duality import enrich_detail

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield

app=FastAPI(title="Polytope5D Search API", version="0.5.0", lifespan=lifespan,
    docs_url=None, redoc_url=None, openapi_url=None)

@app.get("/api/healthz")
def health():
    with connect() as con:
        con.execute("SELECT 1").fetchone()
    return {"ok": True, "version":"0.5.0"}

@app.get("/api/stats")
def stats():
    with connect() as con:
        entries=con.execute("SELECT source,COUNT(*) AS count FROM polytopes GROUP BY source ORDER BY source").fetchall()
    return {"sources":[dict(r) for r in entries],"scope":"local_demo_and_imported_subset_only"}

@app.get("/api/search")
def search(
    source: str | None = Query(None, max_length=100),
    weights: str | None = Query(None, max_length=100),
    reflexive: bool | None = None,
    vertex_min: int | None = Query(None, ge=0, le=1000000),
    vertex_max: int | None = Query(None, ge=0, le=1000000),
    facet_min: int | None = Query(None, ge=0, le=1000000),
    facet_max: int | None = Query(None, ge=0, le=1000000),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=10000),
):
    clauses=[]; args=[]
    for col, value in (("source",source),("reflexive",int(reflexive) if reflexive is not None else None),
                       ("vertex_count",None),("facet_count",None)):
        if value is not None:
            clauses.append(f"{col}=?");args.append(value)
    if weights is not None:
        parts=weights.replace(" ", "").split(",")
        if len(parts)!=6 or any(not p.isdecimal() or int(p)<1 for p in parts):
            raise HTTPException(status_code=422, detail="Enter exactly six positive weights, comma separated")
        clauses.append("weight_key=?")
        args.append(",".join(str(int(p)) for p in parts))
    for col, op, value in (("vertex_count",">=",vertex_min),("vertex_count","<=",vertex_max),
                           ("facet_count",">=",facet_min),("facet_count","<=",facet_max)):
        if value is not None:
            clauses.append(f"{col} {op} ?");args.append(value)
    where=" WHERE "+" AND ".join(clauses) if clauses else ""
    with connect() as con:
        count=con.execute("SELECT COUNT(*) AS c FROM polytopes"+where,args).fetchone()[0]
        # Search returns metadata only. Fetch full geometry individually by ID.
        rows=con.execute("SELECT id,source,dimension,reflexive,weights,weight_key,vertex_count,facet_count,point_count,frequency FROM polytopes"+where+" ORDER BY id LIMIT ? OFFSET ?",args+[limit,offset]).fetchall()
    result=[]
    for r in rows:
        record=dict(r)
        if record["weights"] is not None:
            import json
            record["weights"]=json.loads(record["weights"])
        record.pop("weight_key",None)
        record["reflexive"]=bool(record["reflexive"])
        result.append(record)
    return {"items": result,"total":count,"limit":limit,"offset":offset,"scope":"local_demo_and_imported_subset_only"}

@app.get("/api/polytope/{polytope_id}")
def detail(polytope_id: str):
    if len(polytope_id)>128:
        raise HTTPException(status_code=404,detail="Not found")
    with connect() as con:
        r=con.execute("SELECT * FROM polytopes WHERE id=?",(polytope_id,)).fetchone()
    if r is None:
        raise HTTPException(status_code=404,detail="Not found")
    return enrich_detail(deserialize(r))
