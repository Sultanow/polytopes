import { useEffect, useState } from 'react';
import Polytope4DPoints from './Polytope4DPoints.jsx';
import './catalog4d-live.css';

const SOURCE='https://huggingface.co/datasets/calabi-yau-data/polytopes-4d';
const EMPTY={vertex_count:'',facet_count:'',point_count:'',dual_point_count:'',h11:'',h12:'',euler_characteristic:''};
const FIELDS=[['vertex_count','Vertices'],['facet_count','Facets'],['point_count','Lattice points'],['dual_point_count','Dual lattice points'],['h11','h¹¹'],['h12','h¹²'],['euler_characteristic','Euler χ']];
const LIMIT=12;

function Summary({row}) {
  return <div className="cat4d-numbers">{FIELDS.map(([field,label])=><div key={field}><span>{label}</span><strong>{row?.[field] ?? '–'}</strong></div>)}</div>;
}
export default function Dimension4DLiveExplorer() {
  const [draft,setDraft]=useState(EMPTY),[filters,setFilters]=useState(EMPTY);
  const [offset,setOffset]=useState(0),[results,setResults]=useState([]),[selected,setSelected]=useState(null);
  const [error,setError]=useState(''),[loading,setLoading]=useState(false),[status,setStatus]=useState(null);
  useEffect(()=>{
    fetch('/api/4d/live/status').then(r=>r.ok?r.json():null).then(setStatus).catch(()=>{});
  },[]);
  useEffect(()=>{
    let alive=true;const controller=new AbortController();
    setLoading(true);setError('');
    const qs=new URLSearchParams({offset:String(offset),limit:String(LIMIT)});
    Object.entries(filters).forEach(([k,v])=>{if(v!=='')qs.set(k,v)});
    fetch(`/api/4d/live/search?${qs}`,{signal:controller.signal})
      .then(async r=>{if(!r.ok){let detail='';try{const j=await r.json();detail=j.detail||''}catch{}throw new Error(detail||`HTTP ${r.status}`)}return r.json()})
      .then(data=>{if(alive){setResults(data.items||[]);setSelected(data.items?.[0]||null)}})
      .catch(e=>{if(alive&&e.name!=='AbortError'){setResults([]);setSelected(null);setError(e.message)}})
      .finally(()=>{if(alive)setLoading(false)});
    return ()=>{alive=false;controller.abort()};
  },[filters,offset]);
  const hasFilter=Object.values(filters).some(x=>x!=='');
  return <main className="portal-content cat4d-page">
    <div className="portal-overline">DIMENSION 04 / KREUZER–SKARKE · ON-DEMAND DATA</div>
    <h1>4D reflexive<br/><span>polytopes.</span></h1>
    <p className="portal-lead">Browse real lattice coordinates from the 473,800,776-class classification. Data is requested live from Hugging Face — no local copy of the 15.8 GB catalogue.</p>
    <div className="cat4d-metrics"><div><strong>473,800,776</strong><span>classified 4D polytopes</span></div><div><strong>Remote</strong><span>source: Hugging Face Parquet</span></div><div><strong>{status?.viewer?'Available':status?'Unavailable':'Checking'}</strong><span>Dataset Viewer API</span></div></div>
    <div className="cat4d-warning" role="note"><strong>Coverage warning.</strong> The Hugging Face viewer does not offer full-text search for this dataset. Filter/sort indexes on datasets larger than 5 GB can cover only the first 5 GB. <strong>This is a live pilot, not yet a complete search across all classes.</strong></div>
    <section className="portal-pane cat4d-search">
      <h2>Search remote dataset</h2>
      <form onSubmit={e=>{e.preventDefault();setOffset(0);setFilters({...draft})}}>
        {FIELDS.map(([name,label])=><label key={name}>{label}<input type="number" min={name==='euler_characteristic'?'-1000000':name==='vertex_count'||name==='facet_count'?'5':'0'} max="1000000" step="1" value={draft[name]} placeholder="Any" onChange={e=>setDraft(old=>({...old,[name]:e.target.value}))}/></label>)}
        <div className="cat4d-form-actions"><button type="submit" disabled={loading}>Apply filters</button><button type="button" disabled={loading} onClick={()=>{setDraft(EMPTY);setFilters(EMPTY);setOffset(0)}}>Clear</button></div>
      </form>
      {error&&<div className="cat4d-error" role="alert">Live provider unavailable: {error}. <a href={SOURCE} target="_blank" rel="noreferrer">Open original dataset ↗</a></div>}
      <div className="cat4d-summary" role="status">{loading?'Fetching a small remote page…':`${results.length} records shown ${hasFilter?'from the available filter index':'from the source'}`}</div>
      <div className="cat4d-items">{results.map(row=><button key={row.row_idx} className={selected?.row_idx===row.row_idx?'selected':''} type="button" onClick={()=>setSelected(row)}><b>Row #{row.row_idx.toLocaleString('en-US')}</b><span>{row.vertex_count} vertices · {row.facet_count} facets</span><span>h¹¹={row.h11} · h¹²={row.h12} · χ={row.euler_characteristic}</span></button>)}</div>
      <div className="cat4d-pager"><button type="button" disabled={loading||offset===0} onClick={()=>setOffset(n=>Math.max(0,n-LIMIT))}>← Previous</button><span>Offset {offset.toLocaleString('en-US')}</span><button type="button" disabled={loading||results.length<LIMIT||offset+LIMIT>10000} onClick={()=>setOffset(n=>n+LIMIT)}>Next →</button></div>
    </section>
    <section className="portal-pane cat4d-detail">
      <h2>{selected?`Polytope · source row ${selected.row_idx}`:'4D geometry viewer'}</h2>
      {selected?<>
        <Summary row={selected}/>
        <Polytope4DPoints key={selected.row_idx} vertices={selected.vertices}/>
        <details><summary>Exact 4D integer vertices and metadata (JSON)</summary><pre>{JSON.stringify(selected,null,2)}</pre></details>
        <p className="cat4d-note">The viewer projects true 4D vertices and, where computationally feasible, exact 4D edges into 2D. The wireframe is not a 3D surface mesh; 4D facets are not rendered.</p>
      </>:<p>Select a result to view its exact lattice coordinates.</p>}
    </section>
    <p className="cat4d-attribution">Kreuzer &amp; Skarke · <a href={SOURCE} target="_blank" rel="noreferrer">calabi-yau-data/polytopes-4d</a> · CC BY-SA 4.0. Source row indexes are dataset positions, not permanent mathematical class identifiers. No mass data is stored by this application.</p>
  </main>;
}
