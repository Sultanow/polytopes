import { useCallback, useEffect, useState } from 'react';
import Polytope3DCanvas from './Polytope3DCanvas.jsx';
import './catalog3d.css';

const BASE='/api/3d';
async function get(path) {
  const r=await fetch(`${BASE}${path}`);
  if(!r.ok) throw new Error(`API HTTP ${r.status}`);
  return r.json();
}
export default function Dimension3DExplorer() {
  const [stats,setStats]=useState(null);
  const [filters,setFilters]=useState({vertex_min:'',vertex_max:'',facet_min:'',facet_max:''});
  const [draft,setDraft]=useState({vertex_min:'',vertex_max:'',facet_min:'',facet_max:''});
  const [offset,setOffset]=useState(0);
  const [results,setResults]=useState({items:[],total:0});
  const [selected,setSelected]=useState(null);
  const [jump,setJump]=useState('');
  const [error,setError]=useState('');
  const [loading,setLoading]=useState(false);
  const limit=24;

  const loadDetail=useCallback(async id=>{
    try {setError('');setSelected(await get(`/polytopes/${id}`));}
    catch(e){setError(e.message);}
  },[]);

  useEffect(()=>{get('/stats').then(setStats).catch(e=>setError(`3D API unavailable: ${e.message}`));},[]);
  useEffect(()=>{
    let live=true;
    setLoading(true);
    const qs=new URLSearchParams({limit:String(limit),offset:String(offset)});
    Object.entries(filters).forEach(([k,v])=>{if(v!=='')qs.set(k,v)});
    get(`/polytopes?${qs}`)
      .then(payload=>{if(live){setResults(payload);setError('');}})
      .catch(e=>{if(live)setError(`3D search unavailable: ${e.message}`)})
      .finally(()=>{if(live)setLoading(false)});
    return ()=>{live=false};
  },[filters,offset]);

  return <main className="portal-content cat3d-page">
    <div className="portal-overline">DIMENSION 03 / KREUZER–SKARKE CLASSIFICATION</div>
    <h1>3D reflexive<br/><span>polytopes.</span></h1>
    <p className="portal-lead">Explore the complete classification of 4,319 equivalence classes. Browse exact lattice vertices, their polar duals and validated facet incidences.</p>
    <div className="cat3d-metrics">
      <div><strong>{stats?.count?.toLocaleString('en-US') ?? '—'}</strong><span>loaded classes</span></div>
      <div><strong>4,319</strong><span>classified</span></div>
      <div><strong>{stats?.loaded ? 'Complete' : 'Pending import'}</strong><span>catalogue status</span></div>
    </div>
    {stats && !stats.loaded && <div className="cat3d-notice" role="status">The 3D catalogue has not yet been fully imported. After deployment run <code>docker compose exec api python -m api.catalog3d_import</code> in the application directory on your server. The page never labels an incomplete import as complete.</div>}
    {error && <div className="cat3d-error" role="alert">{error}</div>}
    <section className="portal-pane cat3d-search">
      <h2>Search the catalogue</h2>
      <form onSubmit={e=>{e.preventDefault();setFilters(draft);setOffset(0)}}>
        {[["vertex_min","Vertices ≥"],["vertex_max","Vertices ≤"],["facet_min","Facets ≥"],["facet_max","Facets ≤"]].map(([key,label])=><label key={key}>{label}
          <input type="number" inputMode="numeric" min="4" max="128" value={draft[key]} onChange={e=>setDraft(old=>({...old,[key]:e.target.value}))} placeholder="Any"/>
        </label>)}
        <button type="submit">Apply filters</button>
        <button type="button" className="cat3d-secondary" onClick={()=>{const blank={vertex_min:'',vertex_max:'',facet_min:'',facet_max:''};setDraft(blank);setFilters(blank);setOffset(0)}}>Clear</button>
      </form>
      <div className="cat3d-search-summary">{loading ? 'Searching...' : `${results.total.toLocaleString('en-US')} matching classes`}</div>
      <div className="cat3d-results">
        {results.items.map(item=><button key={item.catalogue_id} type="button" className={selected?.catalogue_id===item.catalogue_id?'active':''} onClick={()=>loadDetail(item.catalogue_id)}>
          <strong>#{item.catalogue_id}</strong><span>{item.vertex_count} vertices · {item.edge_count} edges · {item.facet_count} facets</span>
        </button>)}
      </div>
      <div className="cat3d-pager"><button type="button" disabled={offset===0||loading} onClick={()=>setOffset(Math.max(0,offset-limit))}>Previous</button>
        <span>{results.total ? `${offset+1}–${Math.min(offset+limit,results.total)} of ${results.total}` : '0 results'}</span>
        <button type="button" disabled={loading||offset+limit>=results.total} onClick={()=>setOffset(offset+limit)}>Next</button></div>
    </section>
    <section className="portal-pane cat3d-detail">
      <div className="cat3d-detail-top"><div><span className="portal-small-label">EXACT GEOMETRY</span><h2>{selected ? `Polytope #${selected.catalogue_id}` : 'Select a polytope'}</h2></div>
        <form onSubmit={e=>{e.preventDefault();if(jump!==''&&Number.isInteger(Number(jump))&&Number(jump)>=0&&Number(jump)<=4318)loadDetail(Number(jump));else setError('Enter an index between 0 and 4318.')}}>
          <input type="number" aria-label="SageMath catalogue index" placeholder="Index 0–4318" min="0" max="4318" value={jump} onChange={e=>setJump(e.target.value)}/><button type="submit">Go</button>
        </form>
      </div>
      {selected ? <>
        <div className="cat3d-dual-grid">
          <Polytope3DCanvas key={`p-${selected.catalogue_id}`} title="Δ · Primal polytope" vertices={selected.vertices} edges={selected.edges} facets={selected.facets}/>
          <Polytope3DCanvas key={`d-${selected.catalogue_id}`} title="Δ* · Polar dual" vertices={selected.dual_vertices} edges={selected.dual_edges} facets={selected.dual_facets}/>
        </div>
        <div className="cat3d-fvector">f-vector Δ: ({selected.vertex_count}, {selected.edge_count}, {selected.facet_count}) <span>·</span> f-vector Δ*: ({selected.dual_vertex_count}, {selected.dual_edge_count}, {selected.dual_facet_count})</div>
        <details className="cat3d-coordinates"><summary>Exact lattice coordinates, facets, and normals (JSON)</summary><pre>{JSON.stringify(selected,null,2)}</pre></details>
      </> : <p className="cat3d-muted">Choose a class above to display its actual 3D lattice geometry and polar dual.</p>}
    </section>
    <section className="portal-pane"><h2>Inclusion families · next phase</h2><p>To certify Δᵢ ⊆ Δⱼ up to lattice equivalence, we must find an integral unimodular transformation. Inclusion cannot be inferred from two unrelated stored normal forms. This page currently shows classification and exact polar duality, not an unverified inclusion graph.</p></section>
    <p className="cat3d-attribution">Source: Kreuzer–Skarke 3D classification via SageMath polytopes_db. Original Sage index is zero-based. Source redistribution permissions require clarification before mirroring raw catalogue data in GitHub.</p>
  </main>;
}
