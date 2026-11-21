/**
 * A deliberately narrow, exact arithmetic proof-of-concept.
 * For weights (1,q1,...,q5), with positive integer qi, consider
 * Delta = conv(-q, e1,...,e5) in Z^5.
 * It is reflexive exactly when every qi divides S = 1 + sum(qi).
 * This is NOT a search of the paper's catalogue or a classification up to GL(5,Z).
 */
const unit = (i) => Array.from({ length: 5 }, (_, j) => i === j ? 1 : 0);

export function inspectSimplexWeights(q) {
  if (!Array.isArray(q) || q.length !== 5 || q.some(v => !Number.isSafeInteger(v) || v < 1 || v > 1000)) {
    return { ok: false, reason: 'Enter exactly five integers between 1 and 1000 for q₁,…,q₅.' };
  }
  const sum = 1 + q.reduce((a, b) => a + b, 0);
  const invalid = q.map((v, i) => ({ weight: v, index: i + 1 })).filter(({ weight }) => sum % weight !== 0);
  if (invalid.length) {
    return {
      ok: false,
      sum,
      reason: `Not reflexive in this simplex family: ${invalid.map(({ index, weight }) => `q${index}=${weight}`).join(', ')} does not divide S=${sum}.`,
    };
  }
  const latticeVertices = [q.map(x => -x), ...Array.from({ length: 5 }, (_, i) => unit(i))];
  // Facet normals n satisfy <n,x> >= -1. The first is opposite -q.
  const facetNormals = [Array(5).fill(-1)];
  for (let i = 0; i < 5; i += 1) {
    facetNormals.push(Array.from({ length: 5 }, (_, j) => i === j ? sum / q[i] - 1 : -1));
  }
  return {
    ok: true, sum, weights: [1, ...q], q: [...q], latticeVertices, facetNormals,
    vertexCount: 6, edgeCount: 15, type: '5D reflexive lattice simplex',
  };
}

export function findReflexiveSimplices(maxWeight = 8, limit = 12) {
  const bound = Math.min(12, Math.max(1, Math.trunc(maxWeight)));
  const cap = Math.min(30, Math.max(1, Math.trunc(limit)));
  const results = [];
  // Nondecreasing q removes weight-coordinate permutations, NOT general GL(5,Z) duplicates.
  for (let a = 1; a <= bound; a++)
    for (let b = a; b <= bound; b++)
      for (let c = b; c <= bound; c++)
        for (let d = c; d <= bound; d++)
          for (let e = d; e <= bound; e++) {
            const q = [a, b, c, d, e];
            const s = 1 + a + b + c + d + e;
            if (q.every(w => s % w === 0)) {
              results.push(inspectSimplexWeights(q));
              if (results.length === cap) return results;
            }
          }
  return results;
}

export function toProjectionGeometry(record) {
  if (!record?.ok) throw new Error('A reflexive simplex record is required.');
  const v = record.latticeVertices;
  const center = Array.from({ length: 5 }, (_, j) => v.reduce((acc, p) => acc + p[j], 0) / v.length);
  const scale = Math.max(1, ...v.flatMap(p => p.map((x, j) => Math.abs(x - center[j]))));
  const vertices = v.map(p => p.map((x, j) => 1.2 * (x - center[j]) / scale));
  const edges = [];
  for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) edges.push([i, j]);
  return { vertices, edges };
}
