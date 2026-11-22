import { useEffect, useRef, useState } from 'react';
import Polytope4DPoints from './Polytope4DPoints.jsx';
import './catalog4d-live.css';

const SOURCE='https://huggingface.co/datasets/calabi-yau-data/polytopes-4d';
const EMPTY={vertex_count:'',facet_count:'',point_count:'',dual_point_count:'',h11:'',h12:'',euler_characteristic:''};
const FIELDS=[['vertex_count','Vertices'],['facet_count','Facets'],['point_count','Lattice points'],['dual_point_count','Dual lattice points'],['h11','h¹¹'],['h12','h¹²'],['euler_characteristic','Euler χ']];
const LIMIT=12;
const MAX_ROW=473800775;
function urlRow(){
  if(typeof window==='undefined') return null;
  const value=new URLSearchParams(window.location.search).get('row');
  if(value===null || !/^\d{1,9}$/.test(value)) return null;
  const idx=Number(value);
  return Number.isSafeInteger(idx) && idx<=MAX_ROW ? idx : null;
}
function permalink(idx){
  const link=new URL(window.location.href);
  link.searchParams.set('row',String(idx));
  return link.toString();
}
function clearPermalink(){
  const link=new URL(window.location.href);
  link.searchParams.delete('row');
  window.history.replaceState(window.history.state,'',link.toString());
}


function Summary({row}) {
  return <div className="cat4d-numbers">{FIELDS.map(([field,label])=><div key={field}><span>{label}</span><strong>{row?.[field] ?? '–'}</strong></div>)}</div>;
}
export default function Dimension4DLiveExplorer() {
  const [draft,setDraft]=useState(EMPTY),[filters,setFilters]=useState(EMPTY);
  const [offset,setOffset]=useState(0),[results,setResults]=useState([]),[selected,setSelected]=useState(null);
  const [error,setError]=useState(''),[loading,setLoading]=useState(false),[status,setStatus]=useState(null);
  const [focusedRow,setFocusedRow]=useState(urlRow),[rowInput,setRowInput]=useState(()=>String(urlRow()??''));
  const [rowLoading,setRowLoading]=useState(false),[rowError,setRowError]=useState(''),[copyState,setCopyState]=useState('');
  const [retrySeq,setRetrySeq]=useState(0);
  const focusedRef=useRef(focusedRow);
  focusedRef.current=focusedRow;
  useEffect(()=>{
    const onHistory=()=>{const idx=urlRow();setFocusedRow(idx);setRowInput(String(idx??''));};
    window.addEventListener('popstate',onHistory);
    return ()=>window.removeEventListener('popstate',onHistory);
  },[]);
  useEffect(()=>{
    if(focusedRow===null) return;
    let alive=true;const controller=new AbortController();
    setRowLoading(true);setRowError('');setCopyState('');setSelected(null);
    fetch(`/api/4d/live/row/${focusedRow}`,{signal:controller.signal})
      .then(async r=>{if(!r.ok){let detail='';try{const j=await r.json();detail=j.detail||''}catch{}throw new Error(detail||`HTTP ${r.status}`)}return r.json()})
      .then(data=>{if(alive)setSelected(data.item)})
      .catch(e=>{if(alive&&e.name!=='AbortError')setRowError(e.message)})
      .finally(()=>{if(alive)setRowLoading(false)});
    return ()=>{alive=false;controller.abort()};
  },[focusedRow,retrySeq]);
  function chooseRow(row){
    setFocusedRow(null);setSelected(row);setRowInput(String(row.row_idx));setRowError('');setCopyState('');
    window.history.replaceState(window.history.state,'',permalink(row.row_idx));
  }
  function jumpToRow(e){
    e.preventDefault();
    const value=rowInput.trim();
    const idx=Number(value);
    if(!/^\d{1,9}$/.test(value)||!Number.isSafeInteger(idx)||idx>MAX_ROW){
      setRowError(`Enter a source row between 0 and ${MAX_ROW.toLocaleString('en-US')}.`);return;
    }
    setRowError('');setSelected(null);setCopyState('');
    if(idx===focusedRow){setRetrySeq(n=>n+1);}
    else{setFocusedRow(idx);window.history.pushState({},'',permalink(idx));}
  }
  async function copyPermalink(){
    if(!selected) return;
    const link=permalink(selected.row_idx);
    try{await navigator.clipboard.writeText(link);setCopyState('Link copied');}
    catch{setCopyState(link);}
  }
  function applyFilters(next){setFocusedRow(null);setCopyState('');clearPermalink();setOffset(0);setFilters(next);}
  function changePage(nextOffset){setFocusedRow(null);setCopyState('');clearPermalink();setOffset(nextOffset);}

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
      .then(data=>{if(alive){setResults(data.items||[]);if(focusedRef.current===null){setSelected(data.items?.[0]||null);setCopyState('')}}})
      .catch(e=>{if(alive&&e.name!=='AbortError'){setResults([]);if(focusedRef.current===null)setSelected(null);setError(e.message)}})
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
      <form onSubmit={e=>{e.preventDefault();applyFilters({...draft})}}>
        {FIELDS.map(([name,label])=><label key={name}>{label}<input type="number" min={name==='euler_characteristic'?'-1000000':name==='vertex_count'||name==='facet_count'?'5':'0'} max="1000000" step="1" value={draft[name]} placeholder="Any" onChange={e=>setDraft(old=>({...old,[name]:e.target.value}))}/></label>)}
        <div className="cat4d-form-actions"><button type="submit" disabled={loading}>Apply filters</button><button type="button" disabled={loading} onClick={()=>{setDraft(EMPTY);applyFilters(EMPTY)}}>Clear</button></div>
      </form>
      {error&&<div className="cat4d-error" role="alert">Live provider unavailable: {error}. <a href={SOURCE} target="_blank" rel="noreferrer">Open original dataset ↗</a></div>}
      <div className="cat4d-summary" role="status">{loading?'Fetching a small remote page…':`${results.length} records shown ${hasFilter?'from the available filter index':'from the source'}`}</div>
      <div className="cat4d-items">{results.map(row=><button key={row.row_idx} className={selected?.row_idx===row.row_idx?'selected':''} type="button" onClick={()=>chooseRow(row)}><b>Row #{row.row_idx.toLocaleString('en-US')}</b><span>{row.vertex_count} vertices · {row.facet_count} facets</span><span>h¹¹={row.h11} · h¹²={row.h12} · χ={row.euler_characteristic}</span></button>)}</div>
      <div className="cat4d-pager"><button type="button" disabled={loading||offset===0} onClick={()=>changePage(Math.max(0,offset-LIMIT))}>← Previous</button><span>Offset {offset.toLocaleString('en-US')}</span><button type="button" disabled={loading||results.length<LIMIT||offset+LIMIT>10000} onClick={()=>changePage(offset+LIMIT)}>Next →</button></div>
    </section>
    <section className="portal-pane cat4d-detail">
      <div className="cat4d-jump">
        <div><h2>Open any source row</h2><p>Go directly to a dataset position. No scanning or downloading the catalogue.</p></div>
        <form onSubmit={jumpToRow}>
          <label htmlFor="cat4d-row-input">Source row #</label>
          <input id="cat4d-row-input" type="number" min="0" max={MAX_ROW} step="1" placeholder="e.g. 12345" value={rowInput} onChange={e=>setRowInput(e.target.value)}/>
          <button type="submit" disabled={rowLoading}>Load row</button>
        </form>
      </div>
      {rowLoading&&<p role="status" className="cat4d-note">Loading a single row from Hugging Face…</p>}
      {rowError&&<div className="cat4d-error" role="alert">Direct lookup failed: {rowError}. The provider may restrict access to this position.</div>}
      <h2>{selected?`Polytope · source row ${selected.row_idx}`:'4D geometry viewer'}</h2>
      {selected?<>
        <div className="cat4d-share">
          <a href={permalink(selected.row_idx)} target="_blank" rel="noreferrer">Open direct link ↗</a>
          <button type="button" onClick={copyPermalink}>Copy link</button>
          {copyState&&<span role="status">{copyState}</span>}
        </div>
        <Summary row={selected}/>
        <Polytope4DPoints key={selected.row_idx} vertices={selected.vertices}/>
        <details><summary>Exact 4D integer vertices and metadata (JSON)</summary><pre>{JSON.stringify(selected,null,2)}</pre></details>
        <p className="cat4d-note">The canvas projects exact 4D lattice vertices and certified edge incidences. When all primitive supporting facets are verified, it also displays the exact polar dual. For costly or unverifiable cases, the original point/edge view remains available.</p>
      </>:<p>Select a result to view its exact lattice coordinates.</p>}
    </section>
    <p className="cat4d-attribution">Kreuzer &amp; Skarke · <a href={SOURCE} target="_blank" rel="noreferrer">calabi-yau-data/polytopes-4d</a> · CC BY-SA 4.0. Source row indexes are dataset positions, not permanent mathematical class identifiers. No mass data is stored by this application.</p>
  </main>;
}
