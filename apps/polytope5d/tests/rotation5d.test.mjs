import test from 'node:test';
import assert from 'node:assert/strict';
import {
  edges, fiveDPlanes, identity5D, projectVertex, rotate5D, transform5D, vertices,
} from '../src/geometry.js';

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('All 10 coordinate planes in R5 are present once', () => {
  assert.equal(fiveDPlanes.length, 10);
  assert.equal(new Set(fiveDPlanes.map(p => p.id)).size, 10);
});

test('Hypercube still has 32 vertices and 80 edges', () => {
  assert.equal(vertices.length, 32);
  assert.equal(edges.length, 80);
});

test('90-degree 5D Givens rotation acts in its plane only', () => {
  const matrix = identity5D();
  rotate5D(matrix, 0, 3, Math.PI / 2);
  const point = transform5D([1, 2, 3, 4, 5], matrix);
  close(point[0], -4); close(point[3], 1);
  close(point[1], 2); close(point[2], 3); close(point[4], 5);
});

test('Repeated rotations preserve norms and dot products', () => {
  const m = identity5D();
  for (let k = 0; k < 2000; k += 1) {
    const plane = fiveDPlanes[k % fiveDPlanes.length];
    rotate5D(m, plane.a, plane.b, Math.sin(k) * 0.01);
  }
  const a = [1, 2, 3, 4, 5];
  const b = [-4, 3, 0, 7, 9];
  close(dot(transform5D(a, m), transform5D(a, m)), dot(a, a));
  close(dot(transform5D(a, m), transform5D(b, m)), dot(a, b));
});

test('Projection does not mutate original 5D coordinates', () => {
  const original = [1, 2, 3, 4, 5];
  const m = identity5D();
  rotate5D(m, 0, 4, 0.4);
  const projected = projectVertex(original, 0.5, 500, 300, { yaw: 0.4, pitch: -0.1 }, m);
  assert.deepEqual(original, [1, 2, 3, 4, 5]);
  assert.ok(Number.isFinite(projected.x) && Number.isFinite(projected.y));
});
