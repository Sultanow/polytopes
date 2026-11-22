import { useMemo, useRef, useState } from 'react';

function normalize(vertices) {
  if (!vertices?.length) return [];
  const center = [0, 1, 2].map(i => (Math.min(...vertices.map(v => v[i])) + Math.max(...vertices.map(v => v[i]))) / 2);
  const extent = Math.max(1, ...vertices.flatMap(v => v.map((x, i) => Math.abs(x - center[i]))));
  return vertices.map(v => v.map((x, i) => 1.2 * (x - center[i]) / extent));
}
function rotation(v, yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(pitch), sx = Math.sin(pitch);
  const x = cy*v[0] + sy*v[2], z = -sy*v[0] + cy*v[2];
  return [x, cx*v[1] - sx*z, sx*v[1] + cx*z];
}
function project(v) {
  const factor = 1 / (1 - v[2]*0.085);
  return {x: 320 + v[0]*140*factor, y: 213 - v[1]*140*factor, z: v[2]};
}

export default function Polytope3DCanvas({ vertices = [], edges = [], facets = [], title = '3D polytope' }) {
  const [view, setView] = useState({yaw: -.48, pitch: .3});
  const [facesVisible, setFacesVisible] = useState(true);
  const active = useRef(null);
  const points = useMemo(() => normalize(vertices).map(p => project(rotation(p, view.yaw, view.pitch))), [vertices, view]);
  const polygons = useMemo(() => facets.map((face, i) => ({
    i,
    depth: face.reduce((sum, vertexId) => sum + (points[vertexId]?.z || 0),0)/Math.max(1,face.length),
    points:face.map(vertexId=>points[vertexId]).filter(Boolean).map(p=>`${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' '),
  })).sort((a,b)=>a.depth-b.depth), [facets, points]);
  const onMove = event => {
    if (!active.current) return;
    const dx = event.clientX-active.current.x, dy = event.clientY-active.current.y;
    active.current={x:event.clientX,y:event.clientY};
    setView(prev=>({yaw:prev.yaw+dx*.009,pitch:Math.max(-1.55,Math.min(1.55,prev.pitch+dy*.009))}));
  };
  return <div className="cat3d-viewer">
    <div className="cat3d-viewer-header"><strong>{title}</strong><div>
      <label><input type="checkbox" checked={facesVisible} onChange={e=>setFacesVisible(e.target.checked)}/> Facets</label>
      <button type="button" onClick={()=>setView({yaw:-.48,pitch:.3})}>Reset</button>
    </div></div>
    <svg viewBox="0 0 640 430" className="cat3d-canvas" role="img" aria-label={`${title}, rotate with drag`}
      onPointerDown={e=>{active.current={x:e.clientX,y:e.clientY};e.currentTarget.setPointerCapture(e.pointerId)}}
      onPointerMove={onMove}
      onPointerUp={e=>{active.current=null;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId)}}
      onPointerCancel={()=>{active.current=null}}>
      <g>
        {facesVisible && polygons.map(face=><polygon key={face.i} points={face.points} fill="rgba(78,212,230,0.075)" stroke="rgba(96,234,240,0.18)" strokeWidth="1"/>)}
        {edges.map(([a,b], i)=>points[a]&&points[b]&&<line key={i} x1={points[a].x} y1={points[a].y} x2={points[b].x} y2={points[b].y} stroke="#6de7ef" strokeOpacity=".8" strokeWidth="1.7"/>)}
        {points.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r="4" fill="#ddfbff" stroke="#42cbd7" strokeWidth="1.2"/>)}
      </g>
    </svg>
    <div className="cat3d-viewer-hint">Drag to rotate · {vertices.length} vertices · {edges.length} edges</div>
  </div>;
}
