import { test } from 'node:test';
import { deepStrictEqual, equal, ok } from 'node:assert/strict';
import { inspectSimplexWeights, findReflexiveSimplices, toProjectionGeometry } from './simplexSearch.js';

test('all-one weights yield a reflexive 5-simplex with 6 vertices', () => {
  const r = inspectSimplexWeights([1,1,1,1,1]);
  equal(r.ok, true); equal(r.sum, 6); equal(r.latticeVertices.length, 6); equal(r.facetNormals.length, 6);
  // Every facet normal pairing is >= -1 and exactly -1 for its 5 facet vertices.
  for (const n of r.facetNormals) {
    const pairings = r.latticeVertices.map(v => n.reduce((acc, ni, j) => acc + ni * v[j], 0));
    ok(pairings.every(x => x >= -1));
    equal(pairings.filter(x => x === -1).length, 5);
  }
});

test('nontrivial reflexive weight system has integral polar normals', () => {
  const r = inspectSimplexWeights([1,1,1,2,2]);
  equal(r.ok,true); equal(r.sum, 8); equal(r.facetNormals[4][3], 3); equal(r.facetNormals[5][4], 3);
  const geometry = toProjectionGeometry(r);
  equal(geometry.vertices.length,6); equal(geometry.edges.length,15);
});

test('non-reflexive weight system is rejected exactly', () => {
  const r=inspectSimplexWeights([1,1,1,1,2]);
  equal(r.ok,false); equal(r.sum,7);
});

test('bounded enumeration is sorted, valid and limited', () => {
  const result=findReflexiveSimplices(8,12);
  equal(result.length,12);
  for (const r of result) {
    equal(r.ok,true);
    deepStrictEqual([...r.q].sort((a,b) => a-b),r.q);
    ok(r.q.every(q => r.sum % q === 0));
  }
});
