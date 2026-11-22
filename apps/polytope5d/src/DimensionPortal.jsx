import { useEffect } from 'react';
import FiveDApp from './App.jsx';
import Dimension2DExplorer from './Dimension2DExplorer.jsx';
import NDPreview from './NDPreview.jsx';
import './portal.css';

const root = 'https://polytopes.mathematics.art';
const paper = 'https://www.mdpi.com/2673-9909/6/11/194';
const hugFace = 'https://huggingface.co/datasets/calabi-yau-data/polytopes-4d';
const sage = 'https://github.com/sagemath/data-polytopes';
const dimensions = [
  {dimension:2, count:'16', heading:'The complete polygon family', text:'Exact vertices and the inclusion graph. Explore nested chains in a common lattice.' , status:'Interactive catalogue'},
  {dimension:3, count:'4,319', heading:'A 3D classification', text:'Orbit-ready three-dimensional lattice geometry. Complete data import is next.' , status:'Viewer prototype'},
  {dimension:4, count:'473,800,776', heading:'A vast classification', text:'A searchable index of Kreuzer–Skarke polytopes and their invariants.',status:'Source verified · import next'},
  {dimension:5, count:'Research', heading:'Beyond the classified frontier', text:'Our working 5D explorer, weight-system search, exact dual geometry and research results.', status:'Live explorer'},
];

function route() {
  const p=window.location.pathname.replace(/\/+$/,'') || '/';
  return ['/','/2d','/3d','/4d','/5d','/compute'].includes(p)?p:'/404';
}
function legacy() {
  return window.location.hostname==='polytope5d.mathematics.art' || window.location.pathname.startsWith('/polytope5d');
}
function Nav({active}) {
  return <nav className="portal-nav" aria-label="Polytope dimensions">
    <a className="portal-logo" href={root+'/'}><span className="portal-mark">P<span>.</span></span>POLYTOPES<span className="portal-muted">.MATHEMATICS.ART</span></a>
    <div className="portal-navlinks">{['2d','3d','4d','5d','compute'].map(p=><a key={p} className={active===`/${p}`?'current':''} aria-current={active===`/${p}`?'page':undefined} href={`${root}/${p}`}>{p==='compute'?'Compute':p.toUpperCase()}</a>)}</div>
  </nav>;
}
function Footer(){return <footer className="portal-footer"><span>Polytopes · mathematics.art</span><a href={paper} rel="noreferrer" target="_blank">5D research paper ↗</a><a href="https://github.com/Sultanow/polytopes" rel="noreferrer" target="_blank">GitHub ↗</a></footer>;}
function Home() {
  return <main className="portal-content">
    <div className="portal-overline">AN INTERACTIVE ATLAS OF REFLEXIVE LATTICE POLYTOPES</div>
    <section className="portal-home-hero"><h1>The geometry<br/><span>of possibility.</span></h1><p>Explore classifications from 2D to 4D, study exact lattice inclusion, and investigate the open landscape of five-dimensional reflexive polytopes.</p></section>
    <div className="portal-section-title"><h2>Choose a dimension</h2><p>From sixteen polygons to the frontier of classification.</p></div>
    <div className="portal-dimension-grid">{dimensions.map(d=><a key={d.dimension} className="portal-dimension-card" href={`${root}/${d.dimension}d`}>
      <span className="portal-dimension-top">{String(d.dimension).padStart(2,'0')} / DIMENSIONS <span>↗</span></span>
      <strong className="portal-dimension-number">{d.dimension}<small>D</small></strong>
      <span className="portal-dimension-count">{d.count}</span>
      <h3>{d.heading}</h3><p>{d.text}</p><span className="portal-status">{d.status}</span>
    </a>)}</div>
    <section className="portal-home-feature"><span className="portal-small-label">CENTRAL IDEA</span><h2>Not just individual shapes.<br/><span>Families, inclusion and duality.</span></h2><p>Rather than treating polytopes as disconnected records, trace verified inclusion relations, explore their polar duals, and browse the combinatorial structure of entire classifications.</p><a href={`${root}/2d`}>Start with 16 polygons ↗</a></section>
  </main>;
}
function ThirdFourth({dimension}) {
  const d=dimensions.find(item=>item.dimension===dimension);
  const is4d=dimension===4;
  return <main className="portal-content"><div className="portal-overline">DIMENSION {String(dimension).padStart(2,'0')} / CLASSIFICATION</div>
    <h1>{dimension}D reflexive<br/><span>polytopes.</span></h1><p className="portal-lead">The mathematical classification is complete: <strong>{d.count}</strong> equivalence classes. The interactive catalogue is being connected incrementally.</p>
    <div className="portal-summary"><span><strong>{d.count}</strong> known classes</span><span><strong>{is4d?'4':'3'}</strong> lattice coordinates</span><span><strong>{is4d?'Parquet':'SageMath'}</strong> dataset source</span></div>
    <section className="portal-pane portal-dimension-demo"><div><h2>Interactive projection</h2><p>{is4d?'An actual 4D reflexive simplex from the Hugging Face dataset viewer, projected into 2D.':'The 3D reflexive lattice cube as a verified illustrative example.'} This is a sample viewer, not a completed database browser.</p>
      <NDPreview dimension={dimension}/></div>
      <div className="portal-next-panel"><span className="portal-small-label">NEXT DATA MILESTONE</span><h3>{is4d?'Indexed 4D catalogue':'The 4,319 polytope catalogue'}</h3>
        <p>{is4d?'Offline import of selected vertex-count Parquet shards, searchable metadata and paginated access. Avoid downloading 15.8 GB into the browser.':'Import the complete set of 4,319 reflexive 3D polytopes, index geometric metadata and calculate certified inclusion pairs.'}</p>
        <a href={is4d?hugFace:sage} target="_blank" rel="noreferrer">View original dataset ↗</a>
        {is4d && <p className="portal-fine">Available fields: vertices, vertex_count, facet_count, point_count, dual_point_count, h11, h12, euler_characteristic. License: CC BY-SA 4.0.</p>}
        {!is4d && <p className="portal-fine">Before redistributing the SageMath 3D data in the public site, clarify the dataset redistribution terms with the maintainers.</p>}
      </div></section>
    <section className="portal-pane"><div className="portal-section-title"><h2>Inclusion families</h2><p>Planned: verified relations Δmin ⊆ Δ₁ ⊆ … ⊆ Δmax, with all polytopes aligned to a common lattice.</p></div><p className="portal-fine">Direct comparison of independent normal forms is not a valid inclusion test. For 4D, full pairwise inclusion over hundreds of millions of records would be infeasible; we will index selected families and verified relations.</p></section>
  </main>;
}
function Compute(){return <main className="portal-content"><div className="portal-overline">CITIZEN SCIENCE / FUTURE WORK</div><h1>Collaborative<br/><span>compute.</span></h1><p className="portal-lead">A future distributed research workflow where volunteers can contribute verified computational results to classification problems in higher dimensions.</p><section className="portal-pane"><h2>How it could work</h2><div className="portal-process"><span>01 · Assign bounded tasks</span><span>02 · Compute locally</span><span>03 · Upload compact certificates</span><span>04 · Independently verify</span></div><p className="portal-fine">This page is a concept, not an active contribution or job-distribution service. No executable code from volunteers will run on our server without validation and isolation.</p></section></main>}
export default function DimensionPortal(){
  const path = legacy() ? '/5d' : route();
  useEffect(()=>{document.title=path==='/5d'?'Polytope5D | Polytopes mathematics.art':path==='/'?'Polytopes | mathematics.art':`${path.slice(1).toUpperCase()} | Polytopes mathematics.art`;},[path]);
  if (legacy()) return <FiveDApp/>; // Existing deployed hostname keeps its exact behavior.
  return <div className="portal-root"><Nav active={path}/>{path==='/5d'?<FiveDApp/>:path==='/2d'?<Dimension2DExplorer/>:path==='/3d'?<ThirdFourth dimension={3}/>:path==='/4d'?<ThirdFourth dimension={4}/>:path==='/compute'?<Compute/>:path==='/'?<Home/>:<main className="portal-content"><h1>Page not found</h1><p><a href={root}>Return to the atlas</a></p></main>}{path!=='/5d'&&<Footer/>}</div>;
}
