/**
 * Exact facet, edge, and polar-dual reconstruction for 4D lattice polytopes.
 * Pure BigInt predicates: no floating point in the combinatorial certification.
 * Work is bounded to avoid freezing ordinary browsers; execute in a Web Worker.
 */
const LIMIT = 40;
const MAX_QUADRUPLES = 100_000;
const MAX_FACETS = 1200;
const diff = (a,b) => a.map((x,k)=>x-b[k]);
const dot4 = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
const abs = x => x<0n?-x:x;
function gcd(a,b){a=abs(a);b=abs(b);while(b){[a,b]=[b,a%b]}return a}
function det3(a,b,c) {
  return a[0]*(b[1]*c[2]-b[2]*c[1])
       - a[1]*(b[0]*c[2]-b[2]*c[0])
       + a[2]*(b[0]*c[1]-b[1]*c[0]);
}
function cofactor4(u,v,w) {
  const exclude = col => [u,v,w].map(row=>row.filter((_,i)=>i!==col));
  return [0,1,2,3].map(i=>(i%2===0?1n:-1n)*det3(...exclude(i)));
}
const choose4 = n => n*(n-1)*(n-2)*(n-3)/24;
const failure = reason => ({edges:[],facetCount:0,certified:false,dualAvailable:false,dualVertices:[],dualEdges:[],reason});

function edgesFromFacets(count,facetMasks) {
  const bits=Array.from({length:count},(_,i)=>1n<<BigInt(i));
  const all=(1n<<BigInt(count))-1n, edges=[],degree=Array(count).fill(0);
  for(let a=0;a<count-1;a++) for(let b=a+1;b<count;b++) {
    const pair=bits[a]|bits[b];
    let common=all, found=false;
    for(const facet of facetMasks){ if((facet&pair)===pair){common &=facet;found=true;} }
    if(found&&common===pair){ edges.push([a,b]); degree[a]++; degree[b]++; }
  }
  return {edges,valid:degree.every(x=>x>=4)};
}

/** A negative result NEVER invents dual vertices or Kanten. */
export function exact4DEdges(vertices) {
  if(!Array.isArray(vertices)||vertices.length<5||vertices.length>LIMIT ||
     vertices.some(p=>!Array.isArray(p)||p.length!==4||p.some(x=>!Number.isSafeInteger(x)))) {
    return failure(`Requires 5–${LIMIT} integral 4D vertices`);
  }
  if(choose4(vertices.length)>MAX_QUADRUPLES) return failure('Too many vertices for bounded browser computation');
  const data=vertices.map(p=>p.map(BigInt));
  const n=data.length, bits=Array.from({length:n},(_,i)=>1n<<BigInt(i));
  if(new Set(data.map(p=>p.join(','))).size!==n) return failure('Duplicate vertices in source');
  const facets = new Map();
  let fullDimension=false;
  for(let a=0;a<n-3;a++) for(let b=a+1;b<n-2;b++)
    for(let c=b+1;c<n-1;c++) for(let d=c+1;d<n;d++) {
      const base=data[a];
      let normal=cofactor4(diff(data[b],base),diff(data[c],base),diff(data[d],base));
      if(normal.every(x=>x===0n))continue;
      let hasPos=false,hasNeg=false,mask=0n;
      for(let t=0;t<n;t++){
        const height=dot4(normal,diff(data[t],base));
        if(height>0n)hasPos=true;
        else if(height<0n)hasNeg=true;
        else mask|=bits[t];
        if(hasPos&&hasNeg)break;
      }
      if(hasPos&&hasNeg){fullDimension=true;continue;}
      if(!hasPos&&!hasNeg)continue;
      fullDimension=true;
      const key=mask.toString();
      if(facets.has(key))continue;
      if(facets.size>=MAX_FACETS)return failure('Facet count exceeds browser computation limit');
      if(hasNeg)normal=normal.map(x=>-x); // inward-pointing normal
      const g=normal.reduce((s,x)=>gcd(s,x),0n);
      const primitive=normal.map(x=>x/g);
      const support=-dot4(primitive,base);  // for reflexive Δ this MUST be 1
      facets.set(key,{mask,normal:primitive,support});
    }
  if(!fullDimension||facets.size<5)return failure('Input is not full-dimensional in 4D');
  const facetArray=[...facets.values()];
  const primalGraph=edgesFromFacets(n,facetArray.map(f=>f.mask));
  if(!primalGraph.valid) return {...failure('Source does not contain a complete extreme-vertex list'),facetCount:facets.size};
  const result={edges:primalGraph.edges,facetCount:facets.size,certified:true,
                dualAvailable:false,dualVertices:[],dualEdges:[]};
  if(facetArray.some(f=>f.support!==1n)) {
    return {...result,dualReason:'Primitive facet equations are not all <u,v> ≥ −1 (not verified reflexive)'};
  }
  // Validate all exact polar inequalities and distinct primitive normals.
  for(const facet of facetArray){
    if(data.some(v=>dot4(facet.normal,v)<-1n))return {...result,dualReason:'Polar inequality validation failed'};
  }
  const normalKeys=facetArray.map(f=>f.normal.join(','));
  if(new Set(normalKeys).size!==facetArray.length)return {...result,dualReason:'Duplicate polar vertices'};
  const max=BigInt(Number.MAX_SAFE_INTEGER);
  if(facetArray.some(f=>f.normal.some(x=>abs(x)>max)))return {...result,dualReason:'Polar coordinate exceeds safe browser integer range'};
  const m=facetArray.length;
  const dualFacetMasks = data.map((v,i)=> {
    let mask=0n;
    for(let j=0;j<m;j++){
      if((facetArray[j].mask&bits[i])!==0n)mask |= 1n<<BigInt(j);
    }
    return mask;
  });
  const dualGraph=edgesFromFacets(m,dualFacetMasks);
  if(!dualGraph.valid)return {...result,dualReason:'Polar edge certification failed'};
  return {...result,dualAvailable:true,dualVertices:facetArray.map(f=>f.normal.map(Number)),
    dualEdges:dualGraph.edges};
}
