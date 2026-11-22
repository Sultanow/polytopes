import { useEffect, useState } from 'react';
import Polytope4DPoints from './Polytope4DPoints.jsx';
import { mirrorHodge } from './hodgeMirror4d.mjs';
import { PUBLISHED_SHARDS, FILTER_REQUIRED_SHARDS, signatureOf, anyNumericFilter, summarizeCoverage } from './coverage4d.mjs';
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
  if (!(PUBLISHED_SHARDS.includes(n) && Number.isSafeInteger(i) && i <= 100000000)) return null;
  const fp=p.get('fp'), rev=p.get('rev');
  if (fp && !/^[a-f0-9]{64}$/.test(fp)) return null;
  if (rev && !/^[a-f0-9]{40}$/.test(rev)) return null;
  return {v:n,i,fp:fp||null,rev:rev||null};
}
function permalink(item) {
  const url = new URL(window.location.href);
  url.searchParams.delete('row');
  url.searchParams.delete('shard');
  url.searchParams.delete('shard_row');
  url.searchParams.delete('fp');
  url.searchParams.delete('rev');
  if (item.shard_vertices !== undefined) {
    url.searchParams.set('shard', String(item.shard_vertices));
    url.searchParams.set('shard_row', String(item.shard_row));
    if (item.fingerprint) url.searchParams.set('fp', item.fingerprint);
    if (item.source_revision_pinned && /^[a-f0-9]{40}$/.test(item.source_revision||''))
      url.searchParams.set('rev', item.source_revision);
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

// These are Batyrev mirror predictions for the Calabi–Yau hypersurface,
// NOT a claim that the dual polytope's dataset row has been located.
function HodgeMirror({row}) {
  const mirror=mirrorHodge(row);
  const {h11,h12,chi} = mirror.source || {h11:row.h11,h12:row.h12,chi:row.euler_characteristic};
  const consistent = mirror.valid;
  return <div className="cat4d-mirror" aria-label="Calabi–Yau mirror symmetry">
    <div className="cat4d-mirror-heading"><h3>Mirror symmetry · Hodge numbers</h3>
      <p>For the Calabi–Yau threefold associated with a reflexive 4-polytope, polar duality exchanges h¹¹ and h¹².</p></div>
    {consistent ? <>
      <div className="cat4d-mirror-pair">
        <div className="cat4d-mirror-side"><span>Δ · source</span><strong>({h11}, {h12})</strong><small>χ = {chi}</small></div>
        <div className="cat4d-mirror-arrow" aria-hidden="true">⇄</div>
        <div className="cat4d-mirror-side"><span>Δ* · predicted mirror</span><strong>({h12}, {h11})</strong><small>χ = {-chi}</small></div>
      </div>
      <p className="cat4d-mirror-foot">χ = 2(h¹¹ − h¹²). Predicted mirror invariants, <b>not</b> a verified second catalogue record.</p>
    </> : <p className="cat4d-dual-warning">The source values do not satisfy χ = 2(h¹¹ − h¹²). Mirror values are withheld pending a data check.</p>}
  </div>;
}

export default function Dimension4DLiveExplorer() {
  const [draft,setDraft] = useState(EMPTY);
  const [applied,setApplied] = useState(null);
  // File-row cursors avoid the legacy 10,000-result offset ceiling.
  const [pageCursors,setPageCursors] = useState([null]);
  const [pageIndex,setPageIndex] = useState(0);
  const [nextCursor,setNextCursor] = useState(null);
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
  const [shardList,setShardList] = useState(PUBLISHED_SHARDS);
  const [manifest,setManifest] = useState(null);
  const [manifestError,setManifestError] = useState('');
  // Browser-session only: no extra API storage and no server-wide scan.
  // Keyed by all numeric filters except vertex_count plus the source revision.
  const [coverage,setCoverage] = useState({});
  const revision = manifest?.source_revision || 'main';
  const coverageKey = applied ? signatureOf(applied,revision) : null;
  const entries = coverageKey ? (coverage[coverageKey] || {}) : {};
  const {exactShardCount,allExact,exactCountSubtotal} = summarizeCoverage(shardList,entries);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/4d/remote/shards',{signal:controller.signal}).then(responseJson).then(data=>{
      if (!Array.isArray(data.listed_shards) || !data.listed_shards.every(Number.isInteger))
        throw new Error('Invalid shard manifest');
      setShardList(data.listed_shards);
      setManifest(data);
      setManifestError('');
    }).catch(exc=>{ if(exc.name!=='AbortError') setManifestError('Shard manifest unavailable; showing the locally known list.'); });
    return ()=>controller.abort();
  },[]);

  function markCoverage(filters, v, revisionId, status, count=null) {
    const key=signatureOf(filters,revisionId);
    setCoverage(old=>{
      const current=old[key] || {};
      if (current[v]?.state==='exact' && status!=='exact') return old;
      return {...old,[key]:{...current,[v]:{state:status,...(status==='exact'?{count}:{} )}}};
    });
  }

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
    const linkParams = new URLSearchParams();
    if (shard?.fp) linkParams.set('fp',shard.fp);
    if (shard?.rev) linkParams.set('rev',shard.rev);
    const endpoint = shard
      ? `/api/4d/remote/row/${shard.v}/${shard.i}${linkParams.size ? `?${linkParams}` : ''}`
      : `/api/4d/live/row/${original}`;
    fetch(endpoint,{signal:controller.signal}).then(responseJson).then(data => {
      if (active) setSelected(shard ? normalizeRemote({...data.item,
        source_revision:data.source_revision,source_revision_pinned:data.source_revision_pinned},shard.v) : data.item);
    }).catch(exc => {
      if (active && exc.name !== 'AbortError') setRowError(exc.message);
    }).finally(() => {if (active) setRowLoading(false);});
    return () => {active=false;controller.abort();};
  }, [deepLink,retry]);

  useEffect(() => {
    if (!applied) return;
    let active = true;
    const controller = new AbortController();
    setLoading(true);setError('');setResults([]);setHasMore(false);setNextCursor(null);
    const params = new URLSearchParams({limit:String(LIMIT)});
    if (pageCursors[pageIndex] !== null) params.set('after_row',String(pageCursors[pageIndex]));
    Object.entries(applied).forEach(([name,value]) => {if (value !== '') params.set(name,value);});
    fetch(`/api/4d/remote/search?${params}`,{signal:controller.signal}).then(responseJson).then(data => {
      if (!active) return;
      setResults((data.items||[]).map(item => normalizeRemote({...item,
        source_revision:data.source_revision,source_revision_pinned:data.source_revision_pinned},Number(applied.vertex_count))));
      setHasMore(Boolean(data.has_more));
      setNextCursor(data.next_cursor ?? null);
      markCoverage(applied,Number(applied.vertex_count),data.source_revision || revision,'sampled');
    }).catch(exc => {
      if (active && exc.name !== 'AbortError') {
        setError(exc.message);
        markCoverage(applied,Number(applied.vertex_count),revision,'failed');
      }
    }).finally(() => {if (active) setLoading(false);});
    return () => {active=false;controller.abort();};
  }, [applied,pageIndex,pageCursors]);

  function beginSearch(nextDraft) {
    const v = Number(nextDraft.vertex_count);
    if (!isUInt(String(nextDraft.vertex_count)) || !shardList.includes(v)) {
      setError('Choose one of the published shards (5–33 or 36 vertices).');return;
    }
    if (FILTER_REQUIRED_SHARDS.has(v) && !anyNumericFilter(nextDraft)) {
      setError('For shards with 10–24 vertices, specify at least one additional numeric filter.');return;
    }
    for (const [key] of FIELDS.slice(1)) {
      const val=String(nextDraft[key] ?? '');
      if (val !== '' && (!/^-?\d+$/.test(val) || !Number.isSafeInteger(Number(val)))) {
        setError(`Invalid integer for ${key}.`);return;
      }
    }
    setError('');setExactCount(null);setCountError('');setPageIndex(0);setPageCursors([null]);setNextCursor(null);
    setDeepLink({original:null,shard:null});setSelected(null);
    const clean=new URL(window.location.href);
    ['row','shard','shard_row','fp','rev'].forEach(key=>clean.searchParams.delete(key));
    window.history.replaceState({},'',clean);
    setApplied({...nextDraft});
    const remembered=coverage[signatureOf(nextDraft,revision)]?.[v];
    if (remembered?.state==='exact') setExactCount(remembered.count);
  }
  function applyFilters(event) {
    event.preventDefault();
    beginSearch(draft);
  }
  function chooseCoverageShard(v) {
    const newDraft={...(applied||draft),vertex_count:String(v)};
    setDraft(newDraft);
    if (FILTER_REQUIRED_SHARDS.has(v) && !anyNumericFilter(newDraft)) {
      setError('Enter an additional numeric filter before searching this large shard.');
      return;
    }
    beginSearch(newDraft);
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
    ['shard','shard_row','fp','rev'].forEach(key=>url.searchParams.delete(key));
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
      if (data.count_exact_for_shard !== true || !Number.isSafeInteger(data.count))
        throw new Error('Provider did not confirm an exact shard count');
      setExactCount(data.count);
      markCoverage(applied,Number(applied.vertex_count),data.source_revision || revision,'exact',data.count);
    } catch(exc) {
      setCountError(exc.message);
      markCoverage(applied,Number(applied.vertex_count),revision,'failed');
    }
    finally {setCountLoading(false);}
  }

  return <main className="portal-content cat4d-page">
    <div className="portal-overline">DIMENSION 04 / KREUZER–SKARKE · REMOTE PARQUET</div>
    <h1>4D reflexive<br/><span>polytopes.</span></h1>
    <p className="portal-lead">Search real lattice coordinates without a local copy of the 15.8 GB dataset. Exact numeric filters run remotely against one of 30 published Parquet shards at a time.</p>
    <div className="cat4d-metrics"><div><strong>473,800,776</strong><span>classified 4D polytopes</span></div><div><strong>30</strong><span>published vertex-count shards</span></div><div><strong>40 s</strong><span>per-query worker timeout</span></div></div>
    <div className="cat4d-warning" role="note"><strong>Scope:</strong> Each successful filter checks exactly one selected Parquet shard. The 30 published files span 5–33 and 36 vertices (no 34/35 files). An exact count is shard-local; the full catalogue is <b>not</b> silently scanned. Query memory settings are bounded, but network bytes and process RSS are not hard-capped. Concurrent queries are rejected.</div>

    <section className="portal-pane cat4d-search">
      <h2>Search one 4D Parquet shard</h2>
      <form onSubmit={applyFilters}>
        {FIELDS.map(([name,label]) => <label key={name}>{label}
          {name==='vertex_count' ? <select value={draft.vertex_count}
            onChange={e=>setDraft(d=>({...d,vertex_count:e.target.value}))}>
            {shardList.map(v=><option key={v} value={v}>{v} vertices</option>)}
          </select> : <input type="number" min={name==='euler_characteristic'?'-1000000':'0'}
            max="1000000" step="1" value={draft[name]} placeholder="Any"
            onChange={e => setDraft(d => ({...d,[name]:e.target.value}))}/>}
        </label>)}
        <div className="cat4d-form-actions"><button type="submit" disabled={loading||countLoading}>Search</button>
          <button type="button" disabled={loading||countLoading} onClick={() => {setDraft(EMPTY);setApplied(null);setResults([]);setSelected(null);setExactCount(null);setError('');const clean=new URL(window.location.href);['row','shard','shard_row','fp','rev'].forEach(key=>clean.searchParams.delete(key));window.history.replaceState({},'',clean);}}>Reset</button>
        </div>
      </form>
      {error && <div className="cat4d-error" role="alert">Remote query: {error}</div>}
      <div className="cat4d-summary" role="status">
        {loading?'Querying remote Parquet (up to 40 seconds)…':
          applied ? `${results.length} result(s) on this page · shard ${applied.vertex_count} vertices`:
          'Choose a vertex-count shard and search. For the large central shards, add another numeric filter.'}
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
        <button type="button" disabled={loading||!applied||pageIndex===0} onClick={()=>setPageIndex(i=>i-1)}>← Previous</button>
        <span>Page {pageIndex+1} · cursor-based</span>
        <button type="button" disabled={loading||!applied||!hasMore||nextCursor===null}
          onClick={()=>{setPageCursors(c=>[...c.slice(0,pageIndex+1),nextCursor]);setPageIndex(i=>i+1);}}>Next →</button>
      </div>
    </section>

    <section className="portal-pane cat4d-coverage" aria-labelledby="cat4d-coverage-title">
      <div className="cat4d-coverage-heading"><div>
        <h2 id="cat4d-coverage-title">Search coverage · 30 shards</h2>
        <p>Coverage is tracked for the <b>last submitted filter combination</b>, independently for each vertex count. Click a shard to search it; use “Count all matches” to certify that shard.</p>
      </div><div className="cat4d-coverage-summary" role="status">
        <strong>{exactShardCount} / {shardList.length}</strong><span>exact shard counts</span>
      </div></div>
      {manifestError && <p className="cat4d-note" role="alert">{manifestError}</p>}
      <div className="cat4d-coverage-grid">
        {shardList.map(v=>{
          const entry=entries[v], state=entry?.state || 'untested';
          return <button key={v} type="button"
            disabled={loading||countLoading}
            aria-pressed={applied?.vertex_count===String(v)}
            className={`cat4d-coverage-shard cat4d-coverage-shard--${state}`}
            title={`Shard ${v}: ${state==='exact'?'exact count for current filters':state==='sampled'?'results page only, NOT a complete count':state==='failed'?'last query failed':'not yet queried for these filters'}`}
            onClick={()=>chooseCoverageShard(v)}>
            <b>{v} vertices</b>
            <span>{state==='exact'?`${entry.count.toLocaleString('en-US')} matches`:
              state==='sampled'?'Page loaded':state==='failed'?'Failed / timed out':'Not checked'}</span>
          </button>;
        })}
      </div>
      {allExact && applied ? <p className="cat4d-coverage-total"><b>{exactCountSubtotal.toLocaleString('en-US')} matches across the 30 listed shards.</b> {manifest?.source_revision_pinned
        ? 'All counts use the pinned source revision.'
        : 'Caution: source “main” is mutable; counts made at different times are not guaranteed to refer to a single frozen revision.'}</p>
        : <p className="cat4d-note">{applied ? 'Untested, sampled, failed or timed-out shards are NOT counted as zero. A result page alone does not certify a shard count.' : 'Submit a numeric filter to start measuring coverage. Nothing is scanned automatically.'}</p>}
      <p className="cat4d-note">The manifest lists filenames, not live availability. Counts are held only in this browser session and are cleared on reload. No catalogue-wide background queries are started.</p>
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
        {selected.fingerprint && <p className="cat4d-provenance">Content SHA-256: <code>{selected.fingerprint}</code> · Source: {selected.source_revision_pinned
          ? `pinned ${selected.source_revision.slice(0,12)}` : 'main (mutable; link fingerprint checked on reload)'}</p>}
        <Summary row={selected}/>
        <HodgeMirror row={selected}/>
        <Polytope4DPoints key={identity(selected)} vertices={selected.vertices}/>
        <details><summary>Exact 4D vertices and metadata (JSON)</summary><pre>{JSON.stringify(selected,null,2)}</pre></details>
        <p className="cat4d-note">The canvas projects true 4D vertices and verified edges. Where feasible it also shows the exact polar dual. Source rows and Parquet shard rows are distinct identifiers.</p>
      </> : <p>Choose a search result or open an existing source-row link.</p>}
    </section>
    <p className="cat4d-attribution">Kreuzer &amp; Skarke · <a href={SOURCE} target="_blank" rel="noreferrer">calabi-yau-data/polytopes-4d</a> · CC BY-SA 4.0. No Parquet archives are stored by this app. File-relative row positions may change across dataset revisions.</p>
  </main>;
}
