import { useEffect, useState } from 'react';
import Polytope4DPoints from './Polytope4DPoints.jsx';
import './catalog4d-live.css';

const SOURCE = 'https://huggingface.co/datasets/calabi-yau-data/polytopes-4d';
const FIELDS = [
  ['vertex_count','Vertices'], ['facet_count','Facets'], ['point_count','Lattice points'],
  ['dual_point_count','Dual lattice points'], ['h11','h¹¹'], ['h12','h¹²'],
  ['euler_characteristic','Euler χ'],
];
const EMPTY = {vertex_count:'8',facet_count:'',point_count:'',dual_point_count:'',h11:'',h12:'',euler_characteristic:''};
const LIMIT = 6;
const MAX_ROW = 473800775;
const isUInt = v => /^\d+$/.test(v);
function originalFromUrl() {
  if (typeof window === 'undefined') return null;
  const value = new URLSearchParams(window.location.search).get('row');
  if (!value || !isUInt(value)) return null;
  const n = Number(value);
  return Number.isSafeInteger(n) && n <= MAX_ROW ? n : null;
}
function shardFromUrl() {
  if (typeof window === 'undefined') return null;
  const p = new URLSearchParams(window.location.search);
  const v = p.get('shard'), r = p.get('shard_row');
  if (!v || !r || !isUInt(v) || !isUInt(r)) return null;
  const n = Number(v), i = Number(r);
  return n >= 5 && n <= 15 && Number.isSafeInteger(i) && i <= 100000000 ? {v:n,i} : null;
}
function permalink(item) {
  const url = new URL(window.location.href);
  url.searchParams.delete('row');
  url.searchParams.delete('shard');
  url.searchParams.delete('shard_row');
  if (item.shard_vertices !== undefined) {
    url.searchParams.set('shard', String(item.shard_vertices));
    url.searchParams.set('shard_row', String(item.shard_row));
  } else {
    url.searchParams.set('row', String(item.row_idx));
  }
  return url.toString();
}
async function responseJson(response) {
  const json = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(json.detail || `HTTP ${response.status}`);
  return json;
}
function Summary({row}) {
  return <div className="cat4d-numbers">{FIELDS.map(([field,label]) =>
    <div key={field}><span>{label}</span><strong>{row?.[field] ?? '–'}</strong></div>)}</div>;
}
function identity(item) {
  return item.shard_vertices !== undefined
    ? `Parquet shard ${item.shard_vertices}, row ${item.shard_row.toLocaleString('en-US')}`
    : `HF source row ${item.row_idx.toLocaleString('en-US')}`;
}
function normalizeRemote(item, vertexCount) {
  return {...item, shard_vertices:vertexCount};
}

export default function Dimension4DLiveExplorer() {
  const [draft,setDraft] = useState(EMPTY);
  const [applied,setApplied] = useState(null);
  const [offset,setOffset] = useState(0);
  const [results,setResults] = useState([]);
  const [hasMore,setHasMore] = useState(false);
  const [selected,setSelected] = useState(null);
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState('');
  const [exactCount,setExactCount] = useState(null);
  const [countLoading,setCountLoading] = useState(false);
  const [countError,setCountError] = useState('');
  const [rowInput,setRowInput] = useState(() => String(originalFromUrl() ?? ''));
  const [rowLoading,setRowLoading] = useState(false);
  const [rowError,setRowError] = useState('');
  const [copyState,setCopyState] = useState('');
  const [deepLink,setDeepLink] = useState(() => ({original:originalFromUrl(), shard:shardFromUrl()}));
  const [retry,setRetry] = useState(0);

  useEffect(() => {
    const onPop = () => {
      setDeepLink({original:originalFromUrl(),shard:shardFromUrl()});
      setRowInput(String(originalFromUrl() ?? ''));
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    const original = deepLink.original, shard = deepLink.shard;
    if (original === null && shard === null) return;
    let active = true;
    const controller = new AbortController();
    setRowLoading(true); setRowError(''); setSelected(null);
    const endpoint = shard
      ? `/api/4d/remote/row/${shard.v}/${shard.i}`
      : `/api/4d/live/row/${original}`;
    fetch(endpoint,{signal:controller.signal}).then(responseJson).then(data => {
      if (active) setSelected(shard ? normalizeRemote(data.item,shard.v) : data.item);
    }).catch(exc => {
      if (active && exc.name !== 'AbortError') setRowError(exc.message);
    }).finally(() => {if (active) setRowLoading(false);});
    return () => {active=false;controller.abort();};
  }, [deepLink,retry]);

  useEffect(() => {
    if (!applied) return;
    let active = true;
    const controller = new AbortController();
    setLoading(true);setError('');setResults([]);
    const params = new URLSearchParams({offset:String(offset),limit:String(LIMIT)});
    Object.entries(applied).forEach(([name,value]) => {if (value !== '') params.set(name,value);});
    fetch(`/api/4d/remote/search?${params}`,{signal:controller.signal}).then(responseJson).then(data => {
      if (!active) return;
      setResults((data.items||[]).map(item => normalizeRemote(item,Number(applied.vertex_count))));
      setHasMore(Boolean(data.has_more));
    }).catch(exc => {
      if (active && exc.name !== 'AbortError') setError(exc.message);
    }).finally(() => {if (active) setLoading(false);});
    return () => {active=false;controller.abort();};
  }, [applied,offset]);

  function applyFilters(event) {
    event.preventDefault();
    const v = Number(draft.vertex_count);
    if (!isUInt(draft.vertex_count) || v < 5 || v > 15) {
      setError('Choose one vertex count between 5 and 15.');return;
    }
    if (v >= 10 && Object.entries(draft).filter(([k]) => k !== 'vertex_count').every(([,x]) => x === '')) {
      setError('For 10–15 vertices, specify at least one additional numeric filter.');return;
    }
    setError('');setExactCount(null);setCountError('');setOffset(0);
    setDeepLink({original:null,shard:null});setSelected(null);
    const clean = new URL(window.location.href);
    ['row','shard','shard_row'].forEach(key=>clean.searchParams.delete(key));
    window.history.replaceState({},'',clean);
    setApplied({...draft});
  }
  function chooseItem(item) {
    setRowError('');setCopyState('');setSelected(item);
    window.history.replaceState({},'',permalink(item));
  }
  function gotoRow(event) {
    event.preventDefault();
    const value = rowInput.trim(), idx = Number(value);
    if (!isUInt(value) || !Number.isSafeInteger(idx) || idx > MAX_ROW) {
      setRowError(`Enter a source position between 0 and ${MAX_ROW.toLocaleString('en-US')}.`);return;
    }
    setSelected(null);setRowError('');setCopyState('');
    const next = {original:idx,shard:null};
    if (deepLink.original === idx && !deepLink.shard) setRetry(x => x+1);
    else setDeepLink(next);
    const url = new URL(window.location.href);
    url.searchParams.delete('shard');url.searchParams.delete('shard_row');
    url.searchParams.set('row',String(idx));
    window.history.pushState({},'',url);
  }
  async function copyLink() {
    if (!selected) return;
    const url = permalink(selected);
    try {await navigator.clipboard.writeText(url);setCopyState('Link copied');}
    catch {setCopyState(url);}
  }
  async function fetchExactCount() {
    if (!applied || loading || countLoading) return;
    setExactCount(null);setCountError('');setCountLoading(true);
    const p = new URLSearchParams();
    Object.entries(applied).forEach(([name,value]) => {if(value!=='') p.set(name,value);});
    try {
      const data = await fetch(`/api/4d/remote/count?${p}`).then(responseJson);
      setExactCount(data.count);
    } catch(exc) {setCountError(exc.message);}
    finally {setCountLoading(false);}
  }

  return <main className="portal-content cat4d-page">
    <div className="portal-overline">DIMENSION 04 / KREUZER–SKARKE · REMOTE PARQUET</div>
    <h1>4D reflexive<br/><span>polytopes.</span></h1>
    <p className="portal-lead">Search real lattice coordinates without a local copy of the 15.8 GB dataset. Exact numeric filters run remotely against one Parquet shard at a time.</p>
    <div className="cat4d-metrics"><div><strong>473,800,776</strong><span>classified 4D polytopes</span></div><div><strong>5–15</strong><span>vertex counts in the bounded search pilot</span></div><div><strong>40 s</strong><span>per-query worker timeout</span></div></div>
    <div className="cat4d-warning" role="note"><strong>Scope:</strong> Every successful filter applies to one complete selected shard, not to all 473.8 million classes. 16+ vertex shards are not enabled in this pilot. Query memory settings are bounded, but outbound bytes and process RSS are not hard-capped. Concurrent queries are rejected.</div>

    <section className="portal-pane cat4d-search">
      <h2>Search one 4D Parquet shard</h2>
      <form onSubmit={applyFilters}>
        {FIELDS.map(([name,label]) => <label key={name}>{label}
          <input type="number" min={name==='vertex_count'?'5':name==='euler_characteristic'?'-1000000':'0'}
            max={name==='vertex_count'?'15':'1000000'} step="1" value={draft[name]}
            placeholder={name==='vertex_count'?'Required':'Any'}
            onChange={e => setDraft(d => ({...d,[name]:e.target.value}))}/>
        </label>)}
        <div className="cat4d-form-actions"><button type="submit" disabled={loading||countLoading}>Search</button>
          <button type="button" disabled={loading||countLoading} onClick={() => {setDraft(EMPTY);setApplied(null);setResults([]);setSelected(null);setExactCount(null);setError('');const clean=new URL(window.location.href);['row','shard','shard_row'].forEach(key=>clean.searchParams.delete(key));window.history.replaceState({},'',clean);}}>Reset</button>
        </div>
      </form>
      {error && <div className="cat4d-error" role="alert">Remote query: {error}</div>}
      <div className="cat4d-summary" role="status">
        {loading?'Querying remote Parquet (up to 40 seconds)…':
          applied ? `${results.length} result(s) on this page · shard ${applied.vertex_count} vertices`:
          'Set the filters and search. A vertex count is mandatory.'}
      </div>
      {applied && <div className="cat4d-remote-count">
        <button type="button" disabled={loading||countLoading} onClick={fetchExactCount}>{countLoading?'Counting…':'Count all matches in this shard'}</button>
        {exactCount!==null && <strong>{exactCount.toLocaleString('en-US')} exact matches (shard only)</strong>}
        {countError && <span role="alert">{countError}</span>}
      </div>}
      <div className="cat4d-items">{results.map(row => <button key={`${row.shard_vertices}:${row.shard_row}`}
        className={selected?.shard_vertices===row.shard_vertices && selected?.shard_row===row.shard_row?'selected':''}
        type="button" onClick={()=>chooseItem(row)}>
        <b>Shard {row.shard_vertices} · row #{row.shard_row.toLocaleString('en-US')}</b>
        <span>{row.vertex_count} vertices · {row.facet_count} facets</span>
        <span>h¹¹={row.h11} · h¹²={row.h12} · χ={row.euler_characteristic}</span>
      </button>)}</div>
      <div className="cat4d-pager">
        <button type="button" disabled={loading||!applied||offset===0} onClick={()=>setOffset(Math.max(0,offset-LIMIT))}>← Previous</button>
        <span>Offset {offset.toLocaleString('en-US')}</span>
        <button type="button" disabled={loading||!applied||!hasMore||offset+LIMIT>10000} onClick={()=>setOffset(offset+LIMIT)}>Next →</button>
      </div>
    </section>

    <section className="portal-pane cat4d-detail">
      <div className="cat4d-jump"><div><h2>Open a Hugging Face source row</h2>
        <p>Preserved from v0.11: global source positions are different from Parquet shard rows.</p></div>
        <form onSubmit={gotoRow}><label htmlFor="cat4d-row-input">Global row #</label>
          <input id="cat4d-row-input" type="number" min="0" max={MAX_ROW} value={rowInput} step="1"
            placeholder="e.g. 12345" onChange={e=>setRowInput(e.target.value)}/>
          <button type="submit" disabled={rowLoading}>Load source row</button></form>
      </div>
      {rowLoading && <p className="cat4d-note" role="status">Loading the selected row…</p>}
      {rowError && <div className="cat4d-error" role="alert">Direct lookup failed: {rowError}</div>}
      <h2>{selected ? `Polytope · ${identity(selected)}` : '4D geometry viewer'}</h2>
      {selected ? <>
        <div className="cat4d-share"><a href={permalink(selected)} target="_blank" rel="noreferrer">Open direct link ↗</a>
          <button type="button" onClick={copyLink}>Copy link</button>
          {copyState && <span role="status">{copyState}</span>}</div>
        <Summary row={selected}/>
        <Polytope4DPoints key={identity(selected)} vertices={selected.vertices}/>
        <details><summary>Exact 4D vertices and metadata (JSON)</summary><pre>{JSON.stringify(selected,null,2)}</pre></details>
        <p className="cat4d-note">The canvas projects true 4D vertices and verified edges. Where feasible it also shows the exact polar dual. Source rows and Parquet shard rows are distinct identifiers.</p>
      </> : <p>Choose a search result or open an existing source-row link.</p>}
    </section>
    <p className="cat4d-attribution">Kreuzer &amp; Skarke · <a href={SOURCE} target="_blank" rel="noreferrer">calabi-yau-data/polytopes-4d</a> · CC BY-SA 4.0. No Parquet archives are stored by this app. File-relative row positions may change across dataset revisions.</p>
  </main>;
}
