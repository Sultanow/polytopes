import { useMemo, useState } from 'react';
import HypercubeCanvas from './HypercubeCanvas.jsx';
import CatalogueSearch from './CatalogueSearch.jsx';
import './catalogue.css';
import { findReflexiveSimplices, inspectSimplexWeights, toProjectionGeometry } from './simplexSearch.js';

const articleUrl = 'https://www.mdpi.com/2673-9909/6/11/194';
const MAX_RESULTS = 12;

export default function App() {
  const [playing, setPlaying] = useState(true);
  const [viewMode, setViewMode] = useState('geometry');
  const [maxWeight, setMaxWeight] = useState(8);
  const [results, setResults] = useState(() => findReflexiveSimplices(8, MAX_RESULTS));
  const [selected, setSelected] = useState(null);
  const [manualWeights, setManualWeights] = useState('1, 1, 1, 2, 2');
  const [message, setMessage] = useState('The divisibility criterion is checked using integer arithmetic.');
  const geometry = useMemo(() => selected ? toProjectionGeometry(selected) : null, [selected]);

  const runSearch = () => {
    const found = findReflexiveSimplices(maxWeight, MAX_RESULTS);
    setResults(found);
    setSelected(found[0] ?? null);
    setMessage(`Showing up to ${MAX_RESULTS} examples with 1 ≤ q₁ ≤ … ≤ q₅ ≤ ${maxWeight}. Permutations are suppressed, but GL(5,ℤ)-equivalence is not tested.`);
  };

  const checkManual = (event) => {
    event.preventDefault();
    const values = manualWeights.split(/[\s,;]+/).filter(Boolean).map(x => /^\d+$/.test(x) ? Number(x) : NaN);
    const record = inspectSimplexWeights(values);
    if (record.ok) {
      setSelected(record);
      setMessage(`Reflexive! S=${record.sum}; all qᵢ divide S. The exact lattice vertices and facet normals are available below.`);
    } else {
      setMessage(record.reason);
    }
  };

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="https://mathematics.art/" aria-label="mathematics.art homepage">
          <span className="brand-mark">M<span className="brand-dot">.</span></span>
          <span>mathematics<span className="muted">.art</span></span>
        </a>
        <span className="header-tag"><span className="live-dot" /> RESEARCH PREVIEW · V0.3</span>
      </header>

      <main className="main-content">
        <div className="eyebrow"><span className="eyebrow-line" /> AN EXPERIMENT IN HIGHER DIMENSIONS</div>
        <section className="hero">
          <div className="hero-copy">
            <p className="small-label">POLYTOPE5D / RESEARCH EXPLORER</p>
            <h1>Geometry beyond <span>what we can see.</span></h1>
            <p className="description">Explore five-dimensional lattice polytopes, rotate their projections, and test a mathematically exact family of reflexive simplices.</p>
            <div className="hero-actions">
              <a className="primary-link" href="#search-lab">Explore simplices <span aria-hidden="true">↗</span></a>
              <a className="secondary-link" href={articleUrl} target="_blank" rel="noreferrer">Read the paper ↗</a>
            </div>
          </div>
          <div className="visualization-panel">
            <div className="panel-top"><span>{viewMode === 'geometry' ? 'LIVE 5D → 2D PROJECTION' : 'GEOMETRIC CONCEPT ART'}</span><div className="visual-tabs"><button className={viewMode === 'geometry' ? 'is-active' : ''} onClick={() => setViewMode('geometry')} type="button">Geometry</button><button className={viewMode === 'art' ? 'is-active' : ''} onClick={() => setViewMode('art')} type="button">Artwork</button></div></div>
            {viewMode === 'geometry' ? <HypercubeCanvas playing={playing} geometry={geometry} /> : <img className="hero-art-image" src="./hero-polytope.png" alt="Artistic concept illustration of multidimensional geometry (not a mathematically verified 5D polytope)" />}
            <div className="panel-caption"><span>{viewMode === 'art' ? 'Concept illustration · not an exact 5D polytope' : selected ? 'Reflexive 5D lattice simplex · 6 vertices, 15 edges' : '5D hypercube · 32 vertices, 80 edges'}</span>{viewMode === 'geometry' && <button type="button" onClick={() => setPlaying(v => !v)}>{playing ? 'Pause' : 'Resume'} rotation</button>}</div>
          </div>
        </section>

        <section className="metrics" aria-label="Currently selected polytope">
          <div><span className="metric-label">DIMENSIONS</span><strong>05</strong></div>
          <div><span className="metric-label">VERTICES</span><strong>{selected ? '06' : '32'}</strong></div>
          <div><span className="metric-label">EDGES</span><strong>{selected ? '15' : '80'}</strong></div>
          <div><span className="metric-label">CURRENT STAGE</span><strong className="metric-status">Search demo <span className="metric-star">✦</span></strong></div>
        </section>

        <section id="search-lab" className="search-lab" aria-labelledby="search-title">
          <div className="search-header">
            <span className="small-label">01 / EXACT SEARCH PROTOTYPE</span>
            <h2 id="search-title">Find a reflexive simplex.</h2>
            <p>For positive integers q₁,…,q₅, build Δ = conv(−q, e₁,…,e₅) in ℤ⁵. It is reflexive precisely when every qᵢ divides S = 1 + Σqᵢ. This is an explicitly bounded, specialized family — <strong>not</strong> a search of our published catalogue.</p>
          </div>
          <div className="search-grid">
            <div className="search-card">
              <label htmlFor="weight-limit">Maximum weight: <strong>{maxWeight}</strong></label>
              <input id="weight-limit" type="range" min="2" max="12" step="1" value={maxWeight} onChange={event => setMaxWeight(Number(event.target.value))}/>
              <button className="search-button" type="button" onClick={runSearch}>Search bounded family</button>
              <div className="search-divider" />
              <form onSubmit={checkManual}>
                <label htmlFor="weights">Or test five weights (q₀ = 1 is fixed)</label>
                <input id="weights" className="weight-input" value={manualWeights} onChange={e => setManualWeights(e.target.value)} placeholder="1, 1, 1, 2, 2" spellCheck="false" />
                <button className="search-button secondary-button" type="submit">Check & visualize</button>
              </form>
            </div>
            <div className="search-results">
              <div className="results-label">EXAMPLE RESULTS · {results.length} SHOWN</div>
              <div className="weight-results">
                {results.map((r) => <button type="button" className={`result-chip ${selected?.weights.join(',') === r.weights.join(',') ? 'active' : ''}`} key={r.weights.join(',')} onClick={() => { setSelected(r); setMessage(`Selected weights (${r.weights.join(', ')}), S=${r.sum}.`); }}>({r.weights.join(', ')})</button>)}
              </div>
              <p className="search-message" aria-live="polite">{message}</p>
              {selected && <details className="simplex-details"><summary>Inspect exact lattice data</summary><pre>{JSON.stringify({ weights: selected.weights, sum: selected.sum, vertices: selected.latticeVertices, facet_normals: selected.facetNormals }, null, 2)}</pre></details>}
            </div>
          </div>
          <p className="search-scope">Research scope: the published classification covers Newton polytopes from length-six single weight systems, not all 5D reflexive polytopes. This demo enumerates only a small simplex subfamily, without canonical GL(5,ℤ) deduplication.</p>
        </section>
        <CatalogueSearch />
      </main>
      <footer className="site-footer"><span>© mathematics.art · Polytope5D</span><a href={articleUrl} target="_blank" rel="noreferrer">Research paper ↗</a></footer>
    </div>
  );
}
