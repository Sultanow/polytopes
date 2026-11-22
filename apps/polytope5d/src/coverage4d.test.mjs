import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PUBLISHED_SHARDS, FILTER_REQUIRED_SHARDS, signatureOf,
         anyNumericFilter, summarizeCoverage } from './coverage4d.mjs';

test('30 published shards cover 5–33 and 36, excluding 34 and 35',()=>{
  assert.equal(PUBLISHED_SHARDS.length,30);
  assert.equal(PUBLISHED_SHARDS[0],5);
  assert.equal(PUBLISHED_SHARDS.at(-1),36);
  assert.equal(PUBLISHED_SHARDS.includes(34),false);
  assert.equal(PUBLISHED_SHARDS.includes(35),false);
});

test('filter signature does not depend on selected shard but does on revision',()=>{
  const f1={vertex_count:'8',h11:'101'};
  const f2={vertex_count:'15',h11:'101'};
  assert.equal(signatureOf(f1,'abc'),signatureOf(f2,'abc'));
  assert.notEqual(signatureOf(f1,'abc'),signatureOf(f2,'def'));
  assert.notEqual(signatureOf(f1,'abc'),signatureOf({...f2,h11:'102'},'abc'));
});

test('sampled pages and timeouts do not count as certified zeros',()=>{
  const result=summarizeCoverage([5,6,7],{
    5:{state:'exact',count:0},6:{state:'sampled'},7:{state:'failed'},
  });
  assert.equal(result.exactShardCount,1);
  assert.equal(result.exactCountSubtotal,0);
  assert.equal(result.allExact,false);
});

test('global subtotal is declared complete only if every listed shard is exact',()=>{
  const entries=Object.fromEntries(PUBLISHED_SHARDS.map(v=>[v,{state:'exact',count:v}]));
  const result=summarizeCoverage(PUBLISHED_SHARDS,entries);
  assert.equal(result.exactShardCount,30);
  assert.equal(result.allExact,true);
  assert.equal(result.exactCountSubtotal, PUBLISHED_SHARDS.reduce((a,b)=>a+b,0));
});

test('large central shards require at least one numeric filter',()=>{
  assert.equal(FILTER_REQUIRED_SHARDS.has(10),true);
  assert.equal(FILTER_REQUIRED_SHARDS.has(24),true);
  assert.equal(FILTER_REQUIRED_SHARDS.has(25),false);
  assert.equal(FILTER_REQUIRED_SHARDS.has(36),false);
  assert.equal(anyNumericFilter({vertex_count:'8'}),false);
  assert.equal(anyNumericFilter({vertex_count:'8',h12:'0'}),true);
});
