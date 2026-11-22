/**
 * Exact 4D edge incidence, with no external libraries and no floating-point predicates.
 *
 * A facet is a supporting 3-dimensional hyperplane of the 4D convex hull.
 * Every 4D facet has 4 affinely independent vertices, so enumerating quadruples
 * recovers all facet vertex sets. For two extreme vertices, their minimal common
 * face is the intersection of every facet containing both; it is an edge iff
 * that intersection contains exactly those two vertex indices.
 *
 * BigInt avoids unsafe floating-point determinant comparisons.
 * This module intentionally does NOT construct fictional facets in 3D projection.
 */

const LIMIT = 40;
const MAX_QUADRUPLES = 100_000;

function det3(a, b, c) {
  return a[0]*(b[1]*c[2]-b[2]*c[1])
       - a[1]*(b[0]*c[2]-b[2]*c[0])
       + a[2]*(b[0]*c[1]-b[1]*c[0]);
}
const diff = (a,b) => a.map((x,k)=>x-b[k]);
const dot4 = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];

function cofactor4(u,v,w) {
  const exclude = col => [u,v,w].map(row=>row.filter((_,i)=>i!==col));
  const x0=exclude(0), x1=exclude(1), x2=exclude(2), x3=exclude(3);
  return [det3(...x0),-det3(...x1),det3(...x2),-det3(...x3)];
}
function choose4(n) { return n*(n-1)*(n-2)*(n-3)/24; }

/** @returns {{ edges:number[][], facetCount:number, certified:boolean, reason?:string }} */
export function exact4DEdges(vertices) {
  if (!Array.isArray(vertices) || vertices.length < 5 || vertices.length > LIMIT ||
      vertices.some(p=>!Array.isArray(p)||p.length!==4||p.some(x=>!Number.isSafeInteger(x)))) {
    return {edges:[],facetCount:0,certified:false,reason:`Requires 5–${LIMIT} integral 4D vertices`};
  }
  if (choose4(vertices.length)>MAX_QUADRUPLES) {
    return {edges:[],facetCount:0,certified:false,reason:'Too many vertices for bounded browser computation'};
  }
  const data=vertices.map(p=>p.map(BigInt));
  const n=data.length;
  if (new Set(data.map(p=>p.join(','))).size!==n) {
    return {edges:[],facetCount:0,certified:false,reason:'Duplicate vertices in source'};
  }
  const bits=Array.from({length:n},(_,i)=>1n<<BigInt(i));
  const facets=new Map();
  let fullDimension = false;

  for(let a=0;a<n-3;a++) for(let b=a+1;b<n-2;b++)
    for(let c=b+1;c<n-1;c++) for(let d=c+1;d<n;d++) {
      const base=data[a];
      const normal=cofactor4(diff(data[b],base),diff(data[c],base),diff(data[d],base));
      if (normal.every(v=>v===0n)) continue;
      let hasPos=false,hasNeg=false,mask=0n;
      for(let t=0;t<n;t++) {
        const height=dot4(normal,diff(data[t],base));
        if(height>0n) hasPos=true;
        else if(height<0n) hasNeg=true;
        else mask|=bits[t];
        if(hasPos&&hasNeg) break; // not a supporting hyperplane
      }
      if (hasPos&&hasNeg) { fullDimension=true; continue; }
      if(!hasPos&&!hasNeg) continue; // all vertices lie in the same 3-flat
      fullDimension=true;
      // A supporting 3-flat containing >= 4 vertices is an actual facet.
      // Dedup by exact vertex-incidence mask, not floating normals.
      facets.set(mask.toString(),mask);
    }
  if (!fullDimension || facets.size<5) {
    return {edges:[],facetCount:facets.size,certified:false,reason:'Input is not a full-dimensional 4D polytope'};
  }
  const facetMasks=[...facets.values()];
  const allMask=(1n<<BigInt(n))-1n;
  const edges=[];
  const degree=Array(n).fill(0);
  for(let a=0;a<n-1;a++) for(let b=a+1;b<n;b++) {
    const pair=bits[a]|bits[b];
    let common=allMask;
    let hasCommon=false;
    for(const facet of facetMasks) {
      if((facet&pair)===pair) {common &=facet;hasCommon=true;}
    }
    if(hasCommon && common===pair) {
      edges.push([a,b]); degree[a]++;degree[b]++;
    }
  }
  // Every vertex of a full-dimensional 4-polytope has at least 4 edges.
  if(degree.some(x=>x<4)) {
    return {edges:[],facetCount:facetMasks.length,certified:false,
            reason:'Source points do not form a complete extreme-vertex list'};
  }
  return {edges,facetCount:facetMasks.length,certified:true};
}
