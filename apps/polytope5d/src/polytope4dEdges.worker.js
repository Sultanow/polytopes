import { exact4DEdges } from './polytope4dEdges.mjs';
self.onmessage = event => {
  try {
    self.postMessage(exact4DEdges(event.data.vertices));
  } catch (error) {
    self.postMessage({edges:[],facetCount:0,certified:false,dualAvailable:false,dualVertices:[],dualEdges:[],
      reason:error instanceof Error ? error.message : 'Geometry computation failed'});
  }
};
