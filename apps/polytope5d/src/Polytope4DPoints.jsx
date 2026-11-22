import { useMemo, useRef, useState } from 'react';

const PLANES = [['X','Y',0,1],['X','Z',0,2],['X','W',0,3],['Y','Z',1,2],['Y','W',1,3],['Z','W',2,3]];
const INIT = [0.15, 0.17, 0.54, 0.14, 0.33, 0.22];
function rotate4(v, angles) {
  const p=[...v];
  PLANES.forEach(([, , i,j], k)=>{
    const c=Math.cos(angles[k]), s=Math.sin(angles[k]);
    const a=p[i],b=p[j]; p[i]=c*a-s*b;p[j]=s*a+c*b;
  });
  return p;
}
function project(v, yaw, pitch) {
  const w=1/(1+Math.abs(v[3])*.18);
  const x=v[0]*w,z=v[2]*w,y=v[1]*w;
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const u=x*cy-z*sy, rz=x*sy+z*cy;
  const ry=y*cp-rz*sp, depth=y*sp+rz*cp;
  const p=1/(1+Math.max(-.8,Math.min(.8,depth))*.16);
  return {x:280+u*135*p,y:205-ry*135*p, depth};
}
export default function Polytope4DPoints({vertices}) {
  const [angles,setAngles]=useState(INIT);
  const [camera,setCamera]=useState({yaw:.42,pitch:.20});
  const active=useRef(null);
  const points=useMemo(()=>{
    if(!vertices?.length)return [];
    const center=Array.from({length:4},(_,i)=>(Math.min(...vertices.map(v=>v[i]))+Math.max(...vertices.map(v=>v[i])))/2);
    const scale=Math.max(1,...vertices.flatMap(v=>v.map((value,i)=>Math.abs(value-center[i]))));
    return vertices.map((v,index)=>({index,...project(rotate4(v.map((x,i)=>(x-center[i])/scale),angles),camera.yaw,camera.pitch)}))
      .sort((a,b)=>a.depth-b.depth);
  },[vertices,angles,camera]);
  const reset=()=>{setAngles(INIT);setCamera({yaw:.42,pitch:.20})};
  return <div className="cat4d-visual">
    <div className="cat4d-preview-title"><strong>4D vertex projection → 3D → 2D</strong><button type="button" onClick={reset}>Reset</button></div>
    <svg viewBox="0 0 560 410" role="img" aria-label="Interactive projection of the actual four-dimensional lattice vertices" onPointerDown={e=>{active.current={x:e.clientX,y:e.clientY};e.currentTarget.setPointerCapture(e.pointerId)}} onPointerMove={e=>{if(!active.current)return;const dx=e.clientX-active.current.x,dy=e.clientY-active.current.y;active.current={x:e.clientX,y:e.clientY};setCamera(v=>({yaw:v.yaw+dx*.008,pitch:Math.max(-1.45,Math.min(1.45,v.pitch+dy*.008))}))}} onPointerUp={e=>{active.current=null;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId)}} onPointerCancel={()=>{active.current=null}}>
      <circle cx="280" cy="205" r="152" fill="none" stroke="#306478" strokeDasharray="4 8" opacity=".42" />
      <line x1="75" y1="205" x2="485" y2="205" stroke="#29495c" opacity=".55" />
      <line x1="280" y1="36" x2="280" y2="374" stroke="#29495c" opacity=".55" />
      {points.map(p=><g key={p.index}><circle cx={p.x} cy={p.y} r="7" fill="#39cde1" fillOpacity={.63+Math.max(-.2,Math.min(.25,p.depth*.12))} stroke="#d6fcff" strokeWidth="1.1" /><text x={p.x+10} y={p.y-9} fill="#e7faff" fontSize="11">{p.index}</text></g>)}
    </svg>
    <p className="cat4d-hint">Drag: 3D camera · Sliders: six genuine 4D rotation planes · Points only (no unverified edges)</p>
    <div className="cat4d-planes">{PLANES.map(([a,b],i)=><label key={i}>{a}–{b}
      <input aria-label={`${a}-${b} rotation`} type="range" min="-3.14" max="3.14" step="0.02" value={angles[i]} onChange={e=>setAngles(old=>old.map((v,k)=>k===i?Number(e.target.value):v))} />
    </label>)}</div>
  </div>;
}
