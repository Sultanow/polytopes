import { useMemo, useState } from 'react';
import catalogue from './data/reflexive-2d.json';

const polys = catalogue.polytopes;
const byId = new Map(polys.map(poly => [poly.id, poly]));
const edges = catalogue.cover_edges;
const ccw = 'rgba(107, 226, 229, .9)';

function embed(vertices, [a,b,c,d] = [1,0,0,1]) {
  return vertices.map(([x,y]) => [a*x+b*y,c*x+d*y]);
}
function mul([a,b,c,d],[e,f,g,h]) { return [a*e+b*g,a*f+b*h,c*e+d*g,c*f+d*h]; }
function polyString(vertices, coord) { return vertices.map(v => `${coord(v)[0]},${coord(v)[1]}`).join(' '); }

function LatticeView({ layers, height = 330, id = 'polygon', grid = true }) {
  const vertices = layers.flatMap(l=>l.vertices);
  const span = Math.max(2, ...vertices.flatMap(([x,y])=>[Math.abs(x),Math.abs(y)])) + 0.5;
  const coord = ([x,y]) => [210 + x*175/span, 190 - y*155/span];
  const lines = [];
  for (let x = -Math.floor(span); x <= Math.floor(span); x++) {
    lines.push(<line key={`x${x}`} x1={coord([x,0])[0]} y1="24" x2={coord([x,0])[0]} y2="356"/>);
  }
  for (let y = -Math.floor(span); y <= Math.floor(span); y++) {
    lines.push(<line key={`y${y}`} y1={coord([0,y])[1]} x1="24" y2={coord([0,y])[1]} x2="396"/>);
  }
  return <svg className="portal-lattice" viewBox="0 0 420 380" style={{maxHeight:height}} aria-label={`Exact lattice polygon plot: ${id}`} role="img">
    {grid && <g stroke="#223448" strokeWidth="0.7">{lines}</g>}
    <line x1="24" y1="190" x2="396" y2="190" stroke="#637b91" strokeWidth="0.8"/>
    <line x1="210" y1="24" x2="210" y2="356" stroke="#637b91" strokeWidth="0.8"/>
    {layers.map(({vertices: vs, stroke = ccw, fill = 'rgba(97,223,230,.08)', key},i) =>
      <g key={key ?? i}>
        <polygon points={polyString(vs,coord)} stroke={stroke} fill={fill} strokeWidth={i===0 ? 2.5 : 2} strokeLinejoin="round" />
        {vs.map((v,j) => <circle key={j} cx={coord(v)[0]} cy={coord(v)[1]} r="3.1" fill={stroke} />)}
      </g>
    )}
    <circle cx="210" cy="190" r="4" fill="#f2c77f"/>
    <text x="221" y="181" fontSize="11" fill="#e5c693">0</text>
  </svg>;
}

function findLongestChain() {
  const outgoing = new Map(polys.map(p => [p.id, []]));
  for (const edge of edges) outgoing.get(edge.from).push(edge);
  const cache = new Map();
  const best = id => {
    if (cache.has(id)) return cache.get(id);
    const choices=outgoing.get(id).map(e=>[e,...best(e.to)]);
    const winner=choices.sort((a,b)=>b.length-a.length)[0] ?? [];
    cache.set(id,winner);
    return winner;
  };
  return polys.map(p=>best(p.id)).sort((a,b)=>b.length-a.length)[0];
}
const longest = findLongestChain();

function NestedChain() {
  if (!longest.length) return null;
  const ids = [longest[0].from, ...longest.map(x=>x.to)];
  const mats = ids.map(()=>[1,0,0,1]);
  for (let i=ids.length-2;i>=0;i--) mats[i] = mul(mats[i+1],longest[i].matrix);
  const layers = ids.map((id,i) => ({
    key:id, vertices:embed(byId.get(id).vertices,mats[i]),
    stroke:`hsl(${190 + 175*i/(ids.length-1)},85%,${66+8*i/(ids.length-1)}%)`,
    fill:`hsla(${190 + 175*i/(ids.length-1)},85%,55%,.07)`,
  })).reverse();
  return <div className="portal-chain-grid">
    <div><LatticeView layers={layers} height={350} id="nested inclusion chain" /></div>
    <div>
      <p className="portal-small-label">ONE VERIFIED CHAIN</p>
      <h3>{ids.join(' ⊂ ')}</h3>
      <p>Each inclusion uses its own exact unimodular matrix. We compose these transformations to draw every polygon in the <strong>same lattice coordinate system</strong>.</p>
      <div className="portal-chain-tags">{ids.map((id,i)=><span key={id} style={{borderColor:layers[ids.length-1-i].stroke}}>{id}</span>)}</div>
      <p className="portal-fine">This is a selected longest chain in the computed inclusion graph, not a claim that every polygon belongs to a single nested family.</p>
    </div>
  </div>;
}

function Graph({ selected, setSelected }) {
  const groups = new Map();
  for (const p of polys) {
    if (!groups.has(p.area_twice)) groups.set(p.area_twice,[]);
    groups.get(p.area_twice).push(p);
  }
  const coords=new Map();
  for (const [area,group] of groups) group.forEach((p,i)=>{
    coords.set(p.id,[55+(area-3)*130, 50+(i+1)*360/(group.length+1)]);
  });
  return <div className="portal-graph-scroll"><svg className="portal-graph" viewBox="0 0 900 460" role="img" aria-label="Verified inclusion cover graph of 16 reflexive lattice polygons">
    {edges.map(e=>{const a=coords.get(e.from),b=coords.get(e.to);return <line key={e.from+e.to} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={e.from===selected?'#7de7ea':'#385568'} strokeWidth={e.from===selected?2.5:1.5} />;})}
    {polys.map(p=>{const [x,y]=coords.get(p.id);return <g key={p.id} className="portal-graph-node" role="button" tabIndex="0" onClick={()=>setSelected(p.id)} onKeyDown={ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();setSelected(p.id);}}} aria-label={`Select ${p.id}`}>
      <circle cx={x} cy={y} r="19" fill={selected===p.id?'#86e5e9':'#102b3d'} stroke={selected===p.id?'#d2fdff':'#5b94a9'} strokeWidth="2"/>
      <text x={x} y={y+4} fontSize="11" fill={selected===p.id?'#081421':'#e0f6fa'} textAnchor="middle">{p.id}</text>
    </g>;})}
    {Array.from(groups.keys()).sort((a,b)=>a-b).map(a=><text key={a} x={55+(a-3)*130} y="446" fontSize="12" fill="#799cad" textAnchor="middle">B={a}</text>)}
  </svg></div>;
}

export default function Dimension2DExplorer() {
  const [selected,setSelected]=useState('P01');
  const [target,setTarget]=useState(null);
  const poly=byId.get(selected);
  const outgoing=edges.filter(edge=>edge.from===selected);
  const incoming=edges.filter(edge=>edge.to===selected);
  const active=outgoing.find(e=>e.to===target) ?? outgoing[0] ?? null;
  const displayed=active ? byId.get(active.to) : null;
  const layers=active ? [
    {key:'target',vertices:displayed.vertices,stroke:'#ad9aff',fill:'rgba(173,154,255,.09)'},
    {key:'source',vertices:embed(poly.vertices,active.matrix),stroke:'#6ce7e9',fill:'rgba(108,231,233,.13)'}
  ] : [{key:'source',vertices:poly.vertices}];
  const handleSelect=id=>{setSelected(id);setTarget(null);};
  return <main className="portal-content">
    <div className="portal-overline">DIMENSION 02 / COMPLETE CLASSIFICATION</div>
    <h1>Sixteen polygons.<br/><span>An entire universe.</span></h1>
    <p className="portal-lead">All 16 inequivalent reflexive lattice polygons, with exact integer vertices and a verified inclusion graph. Select a shape to inspect where it sits in the family.</p>
    <div className="portal-summary"><span><strong>16</strong> classes</span><span><strong>{edges.length}</strong> verified cover edges</span><span><strong>ℤ²</strong> exact lattice arithmetic</span></div>
    <section className="portal-pane">
      <div className="portal-section-title"><h2>Browse the classification</h2><p>Click any polygon to inspect its lattice embedding.</p></div>
      <div className="portal-polys-grid">{polys.map(p=>
        <button type="button" key={p.id} className={`portal-poly-tile ${selected===p.id?'active':''}`} onClick={()=>handleSelect(p.id)} aria-pressed={selected===p.id}>
          <svg viewBox="0 0 100 96" aria-hidden="true"><polygon points={p.vertices.map(([x,y])=>`${50+x*11},${46-y*11}`).join(' ')} fill="rgba(111,221,233,.1)" stroke="#86dfe8" strokeWidth="1.5" />{p.vertices.map(([x,y],i)=><circle key={i} cx={50+x*11} cy={46-y*11} r="2" fill="#c4faff"/>)}<circle cx="50" cy="46" r="2" fill="#e9c37a" /></svg>
          <span><strong>{p.id}</strong> · {p.vertex_count} vertices</span>
        </button>
      )}</div>
    </section>
    <section className="portal-pane portal-detail">
      <div><div className="portal-section-title"><h2>{selected} · Inclusion inspector</h2><p>{poly.vertex_count} vertices · normalized area {poly.area_twice/2} · {outgoing.length} outgoing and {incoming.length} incoming verified cover edges.</p></div>
        <LatticeView layers={layers} id={selected} />
        <div className="portal-legend"><span><i className="portal-legend-dot"/> {selected}</span>{displayed&&<span><i className="portal-legend-dot violet"/> {displayed.id}</span>}</div>
      </div>
      <div className="portal-detail-aside"><p className="portal-small-label">IMMEDIATE SUPERSHAPES</p>
        {outgoing.length ? <div className="portal-selection">{outgoing.map(e=><button key={e.to} type="button" className={active?.to===e.to?'chosen':''} onClick={()=>setTarget(e.to)}>{selected} ⊂ {e.to}</button>)}</div> : <p>No larger direct neighbor in this verified graph.</p>}
        <p className="portal-small-label">INCOMING COVER RELATIONS</p>
        {incoming.length ? <div className="portal-selection">{incoming.map(e=><button key={e.from} type="button" onClick={()=>handleSelect(e.from)}>{e.from} ⊂ {selected}</button>)}</div> : <p>No smaller direct neighbor in this verified graph.</p>}
        <details className="portal-exact-details"><summary>Show exact vertex and transformation data</summary><pre>{JSON.stringify({polygon:poly, inclusion:active},null,2)}</pre></details>
        <p className="portal-fine">Containment is checked after a GL(2,ℤ) transformation. Comparing unaligned normal-form coordinates would be mathematically incorrect.</p>
      </div>
    </section>
    <section className="portal-pane"><div className="portal-section-title"><h2>The inclusion graph</h2><p>Edges are verified inclusion relations without an intermediate class in the computed relation. B denotes the boundary lattice-point count.</p></div><Graph selected={selected} setSelected={handleSelect}/></section>
    <section className="portal-pane"><div className="portal-section-title"><h2>From minimal to maximal</h2><p>An exact chain plotted in a single coordinate system.</p></div><NestedChain/></section>
    <p className="portal-source">Generated independently using exact lattice arithmetic; source and regeneration script are bundled in the repository. The classification count 16 is known from the Kreuzer–Skarke/SageMath literature. This graph consists of verified inclusions found by a bounded GL(2,ℤ) search; complete graph certification can follow in a separate step.</p>
  </main>;
}
