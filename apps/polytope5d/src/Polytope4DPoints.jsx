import { useEffect, useRef, useState } from 'react';
import './dualview.css';

const PLANES = [['X','Y',0,1],['X','Z',0,2],['X','W',0,3],['Y','Z',1,2],['Y','W',1,3],['Z','W',2,3]];
const INITIAL_ANGLES = [0.14,0.17,0.48,0.10,0.31,0.23];
function initialView(){return {yaw:.40,pitch:.18,angles:[...INITIAL_ANGLES]}}
const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
function rotate4(input,angles){
  const p=[...input];
  PLANES.forEach(([, , i,j],k)=>{
    const c=Math.cos(angles[k]),s=Math.sin(angles[k]);
    const a=p[i],b=p[j];p[i]=c*a-s*b;p[j]=s*a+c*b;
  });
  return p;
}
function project(v,view,width,height){
  // 4D perspective, followed by 3D orbit and depth shading.
  const w=1/(1+Math.abs(v[3])*.17);
  const x=v[0]*w,y=v[1]*w,z=v[2]*w;
  const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);
  const u=x*cy-z*sy,depth0=x*sy+z*cy;
  const v2=y*cp-depth0*sp,depth=y*sp+depth0*cp;
  const perspective=1/(1+clamp(depth,-.85,.85)*.16);
  const radius=Math.min(width,height)*.29;
  return {x:width/2+u*radius*perspective,y:height/2-v2*radius*perspective,depth};
}

function Canvas4D({vertices,edges,view,tint,showEdges,showNormals,onDrag}){
  const ref=useRef(null),pointer=useRef(null);
  useEffect(()=>{
    const canvas=ref.current;if(!canvas)return;
    const ctx=canvas.getContext('2d');if(!ctx)return;
    const draw=()=>{
      const rect=canvas.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2);
      const width=Math.max(rect.width,1),height=Math.max(rect.height,1);
      const w=Math.max(1,Math.round(width*ratio)),h=Math.max(1,Math.round(height*ratio));
      if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
      ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
      // Restrained reference circle, matching the elegant 5D canvas design.
      ctx.beginPath();ctx.arc(width/2,height/2,Math.min(width,height)*.34,0,Math.PI*2);
      ctx.strokeStyle='rgba(119,199,223,.13)';ctx.lineWidth=.8;ctx.stroke();
      if(!vertices?.length)return;
      const center=Array.from({length:4},(_,k)=>
        (Math.min(...vertices.map(v=>v[k]))+Math.max(...vertices.map(v=>v[k])))/2);
      const scale=Math.max(1,...vertices.flatMap(v=>v.map((a,i)=>Math.abs(a-center[i]))));
      const points=vertices.map(v=>project(rotate4(v.map((a,i)=>(a-center[i])/scale),view.angles),view,width,height));
      if(showNormals){
        ctx.strokeStyle='rgba(252,191,104,.28)';ctx.lineWidth=.9;
        for(const p of points){ctx.beginPath();ctx.moveTo(width/2,height/2);ctx.lineTo(p.x,p.y);ctx.stroke();}
      }
      const color=tint==='violet'?[181,146,255]:[112,220,246];
      if(showEdges){
        const sorted=(edges||[]).filter(([a,b])=>points[a]&&points[b])
          .map(([a,b])=>({a,b,depth:(points[a].depth+points[b].depth)/2}))
          .sort((a,b)=>a.depth-b.depth);
        for(const edge of sorted){
          const a=points[edge.a],b=points[edge.b];
          const alpha=clamp(.30+edge.depth*.13,.20,.78);
          ctx.strokeStyle=`rgba(${color.join(',')},${alpha})`;ctx.lineWidth=edge.depth<0?1:1.35;
          ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
        }
      }
      for(const p of [...points].sort((a,b)=>a.depth-b.depth)){
        ctx.fillStyle=tint==='violet'?'#eee4ff':'#dbfbff';
        ctx.beginPath();ctx.arc(p.x,p.y,clamp(2.1+p.depth*.32,1.5,3),0,Math.PI*2);ctx.fill();
      }
    };
    draw();
    const ro=typeof ResizeObserver!=='undefined'?new ResizeObserver(draw):null;
    ro?.observe(canvas);
    return ()=>ro?.disconnect();
  },[vertices,edges,view,tint,showEdges,showNormals]);
  return <canvas ref={ref} className="cat4d-dual-canvas" role="img" tabIndex={0}
    aria-label={`Interactive 4D ${tint==='violet'?'polar dual':'primal'} wireframe. Drag or use arrow keys to rotate.`}
    style={{cursor:'grab',touchAction:'none'}}
    onPointerDown={e=>{
      if(e.button!==0)return;
      pointer.current={id:e.pointerId,x:e.clientX,y:e.clientY};
      e.currentTarget.setPointerCapture(e.pointerId);
    }}
    onPointerMove={e=>{
      const p=pointer.current;if(!p||p.id!==e.pointerId)return;
      const dx=e.clientX-p.x,dy=e.clientY-p.y;
      p.x=e.clientX;p.y=e.clientY;onDrag(dx,dy);
    }}
    onPointerUp={e=>{pointer.current=null;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId)}}
    onPointerCancel={()=>{pointer.current=null}}
    onLostPointerCapture={()=>{pointer.current=null}}
    onKeyDown={e=>{
      const moves={ArrowLeft:[-12,0],ArrowRight:[12,0],ArrowUp:[0,-12],ArrowDown:[0,12]};
      if(moves[e.key]){e.preventDefault();onDrag(...moves[e.key]);}
    }} />;
}

export default function Polytope4DPoints({vertices}) {
  const [geometry,setGeometry]=useState({edges:[],facetCount:0,certified:false,dualAvailable:false,dualVertices:[],dualEdges:[],loading:true});
  const [linked,setLinked]=useState(true),[swapped,setSwapped]=useState(false),[showEdges,setShowEdges]=useState(true);
  const [showNormals,setShowNormals]=useState(false),[mode,setMode]=useState('orbit'),[selectedSide,setSelectedSide]=useState(0);
  const [shared,setShared]=useState(initialView),[unlinked,setUnlinked]=useState(()=>[initialView(),initialView()]);
  useEffect(()=>{
    const worker=new Worker(new URL('./polytope4dEdges.worker.js',import.meta.url),{type:'module'});
    let alive=true;setGeometry({edges:[],facetCount:0,certified:false,dualAvailable:false,dualVertices:[],dualEdges:[],loading:true});
    const timeout=setTimeout(()=>{
      if(!alive)return;worker.terminate();setGeometry({edges:[],facetCount:0,certified:false,dualAvailable:false,reason:'Geometry calculation timed out'});
    },12000);
    worker.onmessage=e=>{if(!alive)return;clearTimeout(timeout);setGeometry({...e.data,loading:false});worker.terminate();};
    worker.onerror=()=>{if(!alive)return;clearTimeout(timeout);setGeometry({edges:[],facetCount:0,certified:false,dualAvailable:false,reason:'Geometry worker failed'});worker.terminate();};
    worker.postMessage({vertices});
    return ()=>{alive=false;clearTimeout(timeout);worker.terminate();};
  },[vertices]);

  const update=(i,fn)=>{
    if(linked)setShared(fn);
    else setUnlinked(p=>p.map((v,k)=>k===i?fn(v):v));
  };
  const current=i=>linked?shared:unlinked[i];
  const drag=(i,dx,dy)=>update(i,v=>mode==='orbit'
    ? {...v,yaw:v.yaw+dx*.008,pitch:clamp(v.pitch+dy*.008,-1.45,1.45)}
    : {...v,angles:v.angles.map((a,k)=>k===2?a+dx*.008:k===4?a+dy*.008:a)});
  const selectAngle=(k,value)=>update(selectedSide,v=>({...v,angles:v.angles.map((a,j)=>k===j?value:a)}));
  const toggleLink=next=>{
    if(next)setShared({...unlinked[selectedSide],angles:[...unlinked[selectedSide].angles]});
    else setUnlinked([0,1].map(()=>({...shared,angles:[...shared.angles]})));
    setLinked(next);
  };
  const reset=()=>{setShared(initialView());setUnlinked([initialView(),initialView()]);};
  const primal={key:'primal',name:'Δ',description:'Original · integer lattice vertices',vertices,edges:geometry.certified?geometry.edges:[],tint:'cyan'};
  const dual={key:'dual',name:'Δ*',description:'Exact polar · primitive facet normals',vertices:geometry.dualVertices||[],edges:geometry.dualEdges||[],tint:'violet'};
  const panels=swapped?[dual,primal]:[primal,dual];
  const hasDual=!!geometry.dualAvailable;
  return <section className="dualview cat4d-dualview" aria-label="4D polytope and exact polar dual">
    <div className="dualview-header">
      <div><div className="dualview-kicker">EXACT 4D POLAR DUALITY</div>
        <h3>One polytope. Two perspectives.</h3>
        <p>Real 4D coordinates projected to 2D · drag to rotate · six independent 4D planes.</p></div>
      <div className="dualview-toolbar">
        <label className="dualview-toggle"><input type="checkbox" checked={linked} onChange={e=>toggleLink(e.target.checked)}/>Link rotation</label>
        <label className="dualview-toggle"><input type="checkbox" checked={showEdges} onChange={e=>setShowEdges(e.target.checked)}/>True edges</label>
        <label className="dualview-toggle"><input type="checkbox" checked={showNormals} onChange={e=>setShowNormals(e.target.checked)} disabled={!hasDual}/>Normals</label>
        <button type="button" onClick={()=>setSwapped(v=>!v)} disabled={!hasDual}>Swap</button>
        <button type="button" onClick={reset}>Reset</button>
      </div>
    </div>
    <div className="cat4d-rotation-toolbar">
      <div role="group" aria-label="Rotation mode" className="cat4d-mode-buttons">
        <button type="button" className={mode==='orbit'?'active':''} onClick={()=>setMode('orbit')}>3D view</button>
        <button type="button" className={mode==='4d'?'active':''} onClick={()=>setMode('4d')}>True 4D</button>
      </div>
      {!linked&&<label>Adjust
        <select value={selectedSide} onChange={e=>setSelectedSide(Number(e.target.value))}>
          <option value={0}>Left panel</option><option value={1}>Right panel</option>
        </select>
      </label>}
    </div>
    <div className={`dualview-grid ${!hasDual?'cat4d-solo':''}`}>
      {(hasDual?panels:[primal]).map((p,i)=><div className="dualview-panel" key={p.key}>
        <div className="dualview-panel-heading"><strong>{p.name}</strong><span>{p.vertices?.length||0} vertices · {p.edges.length} exact edges</span></div>
        <Canvas4D vertices={p.vertices} edges={p.edges} view={current(i)} tint={p.tint}
          showEdges={showEdges&&geometry.certified} showNormals={showNormals&&p.key==='dual'}
          onDrag={(dx,dy)=>drag(i,dx,dy)}/>
        <p>{p.description}</p>
      </div>)}
    </div>
    {!geometry.loading&&!hasDual&&<p role="status" className="cat4d-dual-warning">The polar dual is not shown: {geometry.dualReason||geometry.reason||'exact dual not available for this record'}. No unverified edges or facets are drawn.</p>}
    <p className="cat4d-hint" role="status">{geometry.loading?'Certifying 4D facet and edge incidence…':geometry.certified?`${geometry.edges.length} exact primal edges · ${geometry.facetCount} supporting facets${hasDual?` · ${geometry.dualEdges.length} exact dual edges`:''}`:geometry.reason||'Geometry unavailable'}</p>
    <div className="cat4d-planes">{PLANES.map(([a,b],i)=><label key={i}>{a}–{b}
      <input aria-label={`${a}-${b} 4D rotation`} type="range" min="-3.14" max="3.14" step="0.02"
        value={current(selectedSide).angles[i]} onChange={e=>selectAngle(i,Number(e.target.value))}/>
    </label>)}</div>
    {hasDual&&<details className="cat4d-dual-details"><summary>Exact polar vertices and edges (JSON)</summary><pre>{JSON.stringify({dual_vertices:geometry.dualVertices,dual_edges:geometry.dualEdges},null,2)}</pre></details>}
    <p className="dualview-footnote">Convention: Δ* = {'{'}u : ⟨u,v⟩ ≥ −1 for every v ∈ Δ{'}'}. All polar coordinates and edge incidences are certified using exact integer arithmetic in a browser worker. Both panels use independent visual scaling; original lattice coordinates are unchanged.</p>
  </section>;
}
