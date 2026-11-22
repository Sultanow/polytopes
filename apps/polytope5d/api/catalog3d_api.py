"""Read-only 3D catalogue API: no public data upload or user-side computation."""
from __future__ import annotations
import json
from fastapi import APIRouter, HTTPException, Query
from .db import connect
from .catalog3d_store import ensure_schema, expand
from .catalog3d_inclusion import ensure_inclusion_schema

router = APIRouter(prefix='/api/3d', tags=['3D'])

@router.get('/stats')
def stats():
    ensure_schema()
    with connect() as con:
        count=con.execute('SELECT COUNT(*) FROM catalogue3d').fetchone()[0]
        provenance={r['name']:r['value'] for r in con.execute('SELECT name,value FROM catalogue3d_provenance')}
        minmax=con.execute('SELECT MIN(vertex_count) v_min,MAX(vertex_count) v_max,MIN(facet_count) f_min,MAX(facet_count) f_max FROM catalogue3d').fetchone()
    return {'loaded':count==4319,'count':count,'expected':4319,'source':provenance.get('source'),
            'sha256':provenance.get('source_file_sha256'),'ranges':dict(minmax),
            'scope':'complete_3d_classification' if count==4319 else 'not_imported_or_partial'}

@router.get('/polytopes')
def search(
    vertex_min:int|None=Query(None,ge=4,le=128),
    vertex_max:int|None=Query(None,ge=4,le=128),
    facet_min:int|None=Query(None,ge=4,le=128),
    facet_max:int|None=Query(None,ge=4,le=128),
    limit:int=Query(24,ge=1,le=100),
    offset:int=Query(0,ge=0,le=4319),
):
    ensure_schema()
    where=[];args=[]
    for col, op, val in [('vertex_count','>=',vertex_min),('vertex_count','<=',vertex_max),
                         ('facet_count','>=',facet_min),('facet_count','<=',facet_max)]:
        if val is not None:
            where.append(f'{col}{op}?');args.append(val)
    w=' WHERE '+' AND '.join(where) if where else ''
    with connect() as con:
        total=con.execute('SELECT COUNT(*) FROM catalogue3d'+w,args).fetchone()[0]
        rows=con.execute('SELECT catalogue_id,vertex_count,edge_count,facet_count FROM catalogue3d'+w+
                         ' ORDER BY catalogue_id LIMIT ? OFFSET ?',args+[limit,offset]).fetchall()
    return {'total':total,'offset':offset,'limit':limit,
            'items':[{'id':f'ks3d-{r["catalogue_id"]:04d}',**dict(r)} for r in rows]}

@router.get('/polytopes/{catalogue_id}')
def detail(catalogue_id:int):
    if not 0<=catalogue_id<=4318:
        raise HTTPException(404,'Index out of range')
    ensure_schema()
    with connect() as con:
        row=con.execute('SELECT * FROM catalogue3d WHERE catalogue_id=?',(catalogue_id,)).fetchone()
    if row is None:
        raise HTTPException(404,'3D record not loaded')
    return expand(row)

@router.get('/inclusions/stats')
def inclusion_stats():
    ensure_inclusion_schema()
    with connect() as con:
        checksum=con.execute("SELECT value FROM catalogue3d_provenance WHERE name='source_file_sha256'").fetchone()
        count=con.execute('SELECT COUNT(*) FROM catalogue3d_inclusion WHERE catalogue_sha256=?',
                          (checksum[0] if checksum else '',)).fetchone()[0]
    return {'certified_relations':count,'method':'signed_coordinate_permutations_only',
            'complete_poset':False,'missing_relations_are_unknown':True}

@router.get('/inclusions/{catalogue_id}')
def inclusions(catalogue_id:int):
    if not 0<=catalogue_id<=4318:
        raise HTTPException(404,'Index out of range')
    ensure_inclusion_schema()
    with connect() as con:
        checksum=con.execute("SELECT value FROM catalogue3d_provenance WHERE name='source_file_sha256'").fetchone()
        all_rows=con.execute('SELECT child_id,parent_id,witness_matrix,method '
                             'FROM catalogue3d_inclusion '
                             'WHERE (child_id=? OR parent_id=?) AND catalogue_sha256=? '
                             'ORDER BY child_id,parent_id',
                             (catalogue_id,catalogue_id,checksum[0] if checksum else '')).fetchall()
    relations=[]
    for row in all_rows:
        relations.append({'child_id':row['child_id'],'parent_id':row['parent_id'],
                          'direction':'contained_in' if row['child_id']==catalogue_id else 'contains',
                          'matrix':json.loads(row['witness_matrix']),
                          'method':row['method']})
    return {'catalogue_id':catalogue_id,'computed':len(relations)>0,
            'complete_poset':False,'missing_relations_are_unknown':True,
            'relations':relations}
