// Demo: 5D hypercube with 2^5 = 32 vertices and 5*2^4 = 80 edges.
// Catalogue geometries supply their own vertices/edges.
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

// Every pair of axes defines one of the 10 independent coordinate planes of R^5.
export const fiveDPlanes = Array.from({ length: 5 }, (_, a) =>
  Array.from({ length: 5 - a - 1 }, (_, offset) => {
    const b = a + offset + 1;
    return { id: `${a}-${b}`, a, b, label: `x${a + 1}–x${b + 1}` };
  }),
).flat();

export const identity5D = () =>
  Array.from({ length: 5 }, (_, i) =>
    Array.from({ length: 5 }, (_, j) => Number(i === j)),
  );

// Left multiply the orientation by a Givens rotation in the (a,b) plane.
// M <- R_ab(theta) M. Thus all rotations preserve lengths and angles.
export function rotate5D(orientation, a, b, radians) {
  if (!Number.isFinite(radians) || a === b || a < 0 || b > 4 || a > 4 || b < 0) {
    throw new Error('Invalid 5D plane rotation');
  }
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  for (let k = 0; k < 5; k += 1) {
    const x = orientation[a][k];
    const y = orientation[b][k];
    orientation[a][k] = c * x - s * y;
    orientation[b][k] = s * x + c * y;
  }
  return orientation;
}

export function transform5D(vertex, orientation) {
  return orientation.map(row => row.reduce((sum, coeff, i) => sum + coeff * vertex[i], 0));
}

function rotatePlane(p, a, b, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const x = p[a];
  const y = p[b];
  p[a] = x * c - y * s;
  p[b] = x * s + y * c;
}

// 5D Givens transform -> pre-existing demo animation -> fixed 5D-to-3D
// linear projection -> mouse camera orbit in 3D -> perspective on the canvas.
// Matrices alter only the displayed coordinates, never the stored lattice vertices.
export function projectVertex(
  vertex, time, width, height,
  view = { yaw: 0, pitch: 0 },
  orientation = null,
) {
  const p = orientation ? transform5D(vertex, orientation) : [...vertex];
  rotatePlane(p, 0, 3, time * 0.23 + 0.15);
  rotatePlane(p, 1, 4, time * 0.17 + 0.8);
  rotatePlane(p, 2, 4, time * 0.13 + 0.32);
  rotatePlane(p, 0, 2, time * 0.11 - 0.1);
  rotatePlane(p, 1, 3, time * 0.09 + 0.3);

  const x = 0.93 * p[0] + 0.37 * p[2] - 0.25 * p[4];
  const y = 0.91 * p[1] - 0.32 * p[3] + 0.17 * p[4];
  const z = 0.58 * p[2] + 0.4 * p[3] + 0.36 * p[4];

  const cy = Math.cos(view.yaw);
  const sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch);
  const sp = Math.sin(view.pitch);
  const rotatedX = x * cy + z * sy;
  const yawZ = z * cy - x * sy;
  const rotatedY = y * cp - yawZ * sp;
  const depth = y * sp + yawZ * cp;

  const perspective = 3.5 / (3.5 - 0.32 * depth);
  const scale = Math.min(width * 0.20, height * 0.25);
  return {
    x: width / 2 + rotatedX * perspective * scale,
    y: height / 2 + rotatedY * perspective * scale,
    depth,
  };
}
