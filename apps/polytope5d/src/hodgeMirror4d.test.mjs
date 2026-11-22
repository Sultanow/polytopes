import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mirrorHodge } from './hodgeMirror4d.mjs';

test('mirror swaps h11 and h12 and reverses Euler characteristic', () => {
  const result = mirrorHodge({ h11: 101, h12: 1, euler_characteristic: 200 });
  assert.deepEqual(result, {
    valid:true,
    source:{ h11:101, h12:1, chi:200 },
    predictedMirror:{ h11:1, h12:101, chi:-200 },
    partnerRowVerified:false,
  });
});

test('mirror involution on Hodge pairs', () => {
  const first = mirrorHodge({h11:27,h12:30,euler_characteristic:-6});
  assert.equal(first.valid, true);
  const next=mirrorHodge({h11:first.predictedMirror.h11,
    h12:first.predictedMirror.h12,euler_characteristic:first.predictedMirror.chi});
  assert.deepEqual(next.predictedMirror, first.source);
});

test('self-mirror Hodge pair has Euler zero', () => {
  const result = mirrorHodge({h11:20,h12:20,euler_characteristic:0});
  assert.equal(result.valid,true);
  assert.equal(result.predictedMirror.chi,0);
});

test('inconsistent source data is flagged, not silently repaired', () => {
  const result=mirrorHodge({h11:101,h12:1,euler_characteristic:202});
  assert.equal(result.valid,false);
  assert.equal(result.reason,'hodge_euler_inconsistent');
});

test('missing/null Hodge numbers are not cast to zero', () => {
  assert.equal(mirrorHodge({h11:null,h12:1,euler_characteristic:0}).valid,false);
  assert.equal(mirrorHodge({h11:3,h12:'1',euler_characteristic:4}).valid,false);
});
