import { useMemo, useRef, useState } from 'react';

const cube3D = Array.from({length:8},(_,i)=>[0,1,2].map(d=>(i & (1<<d))?1:-1));
const cubeEdges = [];
for (let i=0;i<8;i++) for (let d=0;d<3;d++) if (i<(i^(1<<d))) cubeEdges.push([i,i^(1<<d)]);

// Actual first example shown in the calabi-yau-data/polytopes-4d HF dataset viewer.
const sample4D = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,2,3,5],[-2,-3,-4,-5]];
const simplexEdges = Array.from({length:5},(_,i)=>Array.from({length:5-i-1},(_,j)=>[i,i+j+1])).flat();

function rotate(p,a,b,theta){ const q=[...p],c=Math.cos(theta),s=Math.sin(theta); q[a]=c*p[a]-s*p[b];q[b]=s*p[a]+c*p[b];return q; }

export default function NDPreview({dimension=3}) {
  const [angles,setAngles]=useState([.32,-.38]);
  const [planes,setPlanes]=useState(dimension===4?['0-3','1-2']:['0-2','1-2']);
  const pointer=useRef(null);
  const vertices=dimension===4?sample4D:cube3D;
  const edges=dimension===4?simplexEdges:cubeEdges;
  const available=useMemo(()=>Array.from({length:dimension},(_,i)=>Array.from({length:dimension-i-1},(_,j)=>`${i}-${i+j+1}`)).flat(),[dimension]);
  const positions=useMemo(()=>{
    const factor= Math.max(1,...vertices.flatMap(v=>v.map(x=>Math.abs(x))));
    return vertices.map(v=>{
      const normalized=v.map(x=>x/factor);
      const [i,j]=planes[0].split('-').map(Number);const [k,l]=planes[1].split('-').map(Number);
      const p=rotate(rotate(normalized,i,j,angles[0]),k,l,angles[1]);
      const x=.95*p[0]+.2*(p[3]||0), y=.9*p[1]-.18*(p[3]||0), z=.85*p[2]+.3*(p[3]||0);
      const perspective=3.3/(3.3-z*.3);
      return {x:240+135*x*perspective,y:178-132*y*perspective,z};
    });
  },[dimension,angles,planes]);
  const onMove=event=>{
    if (!pointer.current || pointer.current.id!==event.pointerId) return;
    const dx=event.clientX-pointer.current.x,dy=event.clientY-pointer.current.y;
    pointer.current.x=event.clientX;pointer.current.y=event.clientY;
    setAngles(old=>[old[0]+dx*.008,old[1]+dy*.008]);
  };
  const onEnd=event=>{
    if (pointer.current?.id!==event.pointerId)return;
    pointer.current=null;
    if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return <div className="portal-nd-view">
    <div className="portal-nd-top"><span>LIVE {dimension}D PROJECTION</span><div className="portal-nd-planes">{planes.map((p,index)=><label key={index}>{index===0?'↔':'↕'} <select value={p} onChange={e=>setPlanes(current=>current.map((value,i)=>i===index?e.target.value:value))} aria-label={`${index===0?'Horizontal':'Vertical'} rotation plane`}>
      {available.map(value=><option key={value} value={value}>{value.replace(/(\d+)-(\d+)/,(_,a,b)=>`x${Number(a)+1}–x${Number(b)+1}`)}</option>)}
    </select></label>)}</div></div>
    <svg viewBox="0 0 480 356" role="img" aria-label={`Interactive projection of an actual ${dimension}D lattice polytope. Drag to rotate.`} tabIndex="0" className="portal-nd-svg" style={{touchAction:'none',cursor:'grab'}}
      onPointerDown={event=>{if(event.button!==0)return;event.preventDefault();pointer.current={id:event.pointerId,x:event.clientX,y:event.clientY};event.currentTarget.setPointerCapture(event.pointerId);}}
      onPointerMove={onMove} onPointerUp={onEnd} onPointerCancel={onEnd} onLostPointerCapture={onEnd}
      onKeyDown={event=>{const d={ArrowLeft:[-.12,0],ArrowRight:[.12,0],ArrowUp:[0,-.12],ArrowDown:[0,.12]}[event.key];if(d){event.preventDefault();setAngles(a=>[a[0]+d[0],a[1]+d[1]]);}}}>
      <g opacity=".5"><path d="M30 178 H450 M240 24 V334" stroke="#3e5969" strokeWidth=".8"/></g>
      {edges.map(([a,b],i)=><line key={i} x1={positions[a].x} y1={positions[a].y} x2={positions[b].x} y2={positions[b].y} stroke="#81dce8" strokeWidth="2" opacity={.6+(positions[a].z+positions[b].z)*.12}/>)}
      {positions.map((p,i)=><circle key={i} cx={p.x} cy={p.y} r="4.2" fill="#defbff"/>)}
    </svg>
    <p className="portal-nd-hint">Drag to rotate in the chosen coordinate planes. Lattice coordinates are unchanged.</p>
  </div>;
}
