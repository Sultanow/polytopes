import { useEffect, useMemo, useState } from 'react';
import HypercubeCanvas from './HypercubeCanvas.jsx';

// Relative URL works on both polytope5d.mathematics.art/ and mathematics.art/polytope5d/.
function api(path) { return new URL(`./api/${path}`, document.baseURI).toString(); }

function geometryFromRecord(record) {
  if (!Array.isArray(record?.vertices_5d) || record.vertices_5d.length === 0) return null;
  const points = record.vertices_5d;
  const centroid = Array.from({ length: 5 }, (_, i) => points.reduce((sum, v) => sum + v[i], 0) / points.length);
  const extent = Math.max(1, ...points.flatMap(p => p.map((v, i) => Math.abs(v - centroid[i]))));
  return {
    vertices: points.map(v => v.map((x, i) => (x - centroid[i]) * 1.2 / extent)),
    // Never fabricate the 1-skeleton: show only points when the catalogue has no edges.
    edges: Array.isArray(record.edges) ? record.edges : [],
  };
}

export default function CatalogueSearch() {
  const [sources, setSources] = useState([]);
  const [source, setSource] = useState('');
  const [weights, setWeights] = useState('');
  const [vmin, setVmin] = useState('');
  const [fmax, setFmax] = useState('');
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(null);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [page, setPage] = useState(0);
  const limit = 20;

  async function search(query, nextPage = 0) {
    setLoading(true);
    setError('');
    const params = new URLSearchParams({ limit: String(limit), offset: String(nextPage * limit), reflexive: 'true' });
    if (query.source) params.set('source', query.source);
    if (query.weights.trim()) params.set('weights', query.weights.trim());
    if (query.vmin !== '') params.set('vertex_min', query.vmin);
    if (query.fmax !== '') params.set('facet_max', query.fmax);
    try {
      const response = await fetch(api(`search?${params}`));
      if (!response.ok) {
        const failure = await response.json().catch(() => ({}));
        throw new Error(typeof failure.detail === 'string' ? failure.detail : `API error ${response.status}`);
      }
      const payload = await response.json();
      setResults(payload.items);
      setTotal(payload.total);
      setPage(nextPage);
      setSelected(null);
    } catch (err) {
      setResults([]); setTotal(null);
      setError(`${err.message}. Check whether the API container is running.`);
    } finally { setLoading(false); }
  }

  useEffect(() => {
    const controller = new AbortController();
    fetch(api('stats'), { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error('API unavailable'); return r.json(); })
      .then(data => setSources(data.sources))
      .catch(() => {});
    search({ source: '', weights: '', vmin: '', fmax: '' });
    return () => controller.abort();
  }, []);

  async function showDetails(item) {
    setError('');
    try {
      const response = await fetch(api(`polytope/${encodeURIComponent(item.id)}`));
      if (!response.ok) throw new Error(`Detail request failed (${response.status})`);
      setSelected(await response.json());
    } catch (err) { setError(err.message); }
  }

  const geometry = useMemo(() => geometryFromRecord(selected), [selected]);
  const params = {source, weights, vmin, fmax};
  const isDemoOnly = sources.length > 0 && sources.every(s => s.source === 'analytic_demo');

  return <section id="catalogue-lab" className="catalogue-lab" aria-labelledby="catalogue-title">
    <p className="small-label">02 / SEARCH API · V0.3</p>
    <h2 id="catalogue-title">Browse the indexed sample.</h2>
    <p className="catalogue-intro">Query a real SQLite index via FastAPI. The initial {isDemoOnly ? '69' : 'local'} examples are <strong>analytically constructed demonstrations</strong>, not records from the published multi-billion-polytope classification. A curated catalogue excerpt can be imported separately.</p>
    <form className="catalogue-form" onSubmit={e => { e.preventDefault(); search(params); }}>
      <label>Source
        <select value={source} onChange={e => setSource(e.target.value)}>
          <option value="">All local sources</option>
          {sources.map(s => <option value={s.source} key={s.source}>{s.source} ({s.count})</option>)}
        </select>
      </label>
      <label>Six weights (exact)
        <input value={weights} onChange={e => setWeights(e.target.value)} placeholder="1,1,1,1,2,2" />
      </label>
      <label>Vertices ≥
        <input type="number" min="0" step="1" value={vmin} onChange={e => setVmin(e.target.value)} placeholder="e.g. 10" />
      </label>
      <label>Facets ≤
        <input type="number" min="0" step="1" value={fmax} onChange={e => setFmax(e.target.value)} placeholder="e.g. 32" />
      </label>
      <button type="submit" disabled={loading}>Search index</button>
    </form>
    {error && <p className="catalogue-error" role="alert">{error}</p>}
    <div className="catalogue-layout">
      <div className="catalogue-records">
        <div className="catalogue-records-head">{loading ? 'Searching…' : `${total === null ? '–' : total.toLocaleString('en-US')} matching records`}</div>
        {results.length === 0 && !loading && <p>No matching records in this local subset.</p>}
        {results.map(r => <button className={`catalogue-record ${selected?.id === r.id ? 'active' : ''}`} key={r.id} onClick={() => showDetails(r)} type="button">
          <span className="catalogue-record-id">{r.id}</span>
          <span className="catalogue-record-numbers">V: {r.vertex_count ?? '—'} · F: {r.facet_count ?? '—'} {r.weights ? `· q: (${r.weights.join(',')})` : ''}</span>
        </button>)}
        <div className="catalogue-pager">
          <button type="button" disabled={loading || page === 0} onClick={() => search(params, page - 1)}>Previous</button>
          <span>{total === null ? '—' : `${page * limit + (total ? 1 : 0)}–${Math.min((page + 1) * limit, total)} of ${total}`}</span>
          <button type="button" disabled={loading || total === null || (page + 1) * limit >= total} onClick={() => search(params, page + 1)}>Next</button>
        </div>
      </div>
      <div className="catalogue-detail">
        {!selected ? <p>Choose a search result to inspect its coordinates and 5D projection.</p> : <>
          <div className="catalogue-detail-title"><strong>{selected.id}</strong><span>{selected.source}</span></div>
          <div className="catalogue-properties">
            <span>Dimension <b>5</b></span><span>Vertices <b>{selected.vertex_count ?? '—'}</b></span>
            <span>Facets <b>{selected.facet_count ?? '—'}</b></span><span>Points <b>{selected.point_count ?? '—'}</b></span>
            <span>Reflexive <b>{selected.reflexive ? 'Yes' : 'No'}</b></span><span>Frequency <b>{selected.frequency ?? '—'}</b></span>
          </div>
          {geometry ? <div className="catalogue-preview">
            <HypercubeCanvas playing geometry={geometry}/>
            <span>{selected.edges ? `Exact stored graph: ${selected.edges.length} edges` : 'Only stored vertices; no edge graph provided'}</span>
          </div> : <p>No 5D vertex coordinates in this record. Metadata is searchable; a geometric reconstruction would require additional catalogue data.</p>}
          <details className="simplex-details"><summary>Inspect complete JSON record</summary><pre>{JSON.stringify(selected,null,2)}</pre></details>
        </>}
      </div>
    </div>
    <p className="search-scope">This is a staging search index, <strong>not</strong> the published catalogue. Records from an actual catalogue subset must retain their source, stable ID, and exact coordinates where available. Similar metadata does not establish GL(5,ℤ)-equivalence.</p>
  </section>;
}
