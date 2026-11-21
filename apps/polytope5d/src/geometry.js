// Demo: 5-dimensionaler Hyperwürfel mit 2^5 = 32 Ecken und 5*2^4 = 80 Kanten.
// Keine behauptete Darstellung eines Datensatz-Polytops.
export const vertices = Array.from({ length: 32 }, (_, index) =>
  Array.from({ length: 5 }, (_, dimension) =>
    (index & (1 << dimension)) === 0 ? -1 : 1,
  ),
);

export const edges = [];
for (let i = 0; i < 32; i += 1) {
  for (let dimension = 0; dimension < 5; dimension += 1) {
    const j = i ^ (1 << dimension);
    if (i < j) edges.push([i, j]);
  }
}

function rotatePlane(p, a, b, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const x = p[a];
  const y = p[b];
  p[a] = x * c - y * s;
  p[b] = x * s + y * c;
}

export function projectVertex(vertex, time, width, height) {
  const p = [...vertex];
  rotatePlane(p, 0, 3, time * 0.23 + 0.15);
  rotatePlane(p, 1, 4, time * 0.17 + 0.8);
  rotatePlane(p, 2, 4, time * 0.13 + 0.32);
  rotatePlane(p, 0, 2, time * 0.11 - 0.1);
  rotatePlane(p, 1, 3, time * 0.09 + 0.3);

  // Linearer 5D -> 3D Blick, danach perspektivische 3D -> 2D Projektion.
  const x = 0.93 * p[0] + 0.37 * p[2] - 0.25 * p[4];
  const y = 0.91 * p[1] - 0.32 * p[3] + 0.17 * p[4];
  const depth = 0.58 * p[2] + 0.4 * p[3] + 0.36 * p[4];
  const perspective = 3.5 / (3.5 - 0.32 * depth);
  const scale = Math.min(width * 0.20, height * 0.25);

  return {
    x: width / 2 + x * perspective * scale,
    y: height / 2 + y * perspective * scale,
    depth,
  };
}
