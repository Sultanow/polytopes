// Demo: 5D hypercube with 2^5 = 32 vertices and 5*2^4 = 80 edges.
// For catalogue geometries the vertices/edges can be passed in separately.
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

// view = a mouse-controlled 3D orbit (yaw/pitch) applied AFTER 5D -> 3D.
// Keeping time and view separate means the auto-rotation can pause while dragging.
export function projectVertex(vertex, time, width, height, view = { yaw: 0, pitch: 0 }) {
  const p = [...vertex];
  rotatePlane(p, 0, 3, time * 0.23 + 0.15);
  rotatePlane(p, 1, 4, time * 0.17 + 0.8);
  rotatePlane(p, 2, 4, time * 0.13 + 0.32);
  rotatePlane(p, 0, 2, time * 0.11 - 0.1);
  rotatePlane(p, 1, 3, time * 0.09 + 0.3);

  // First, the 5D -> 3D linear projection.
  const x = 0.93 * p[0] + 0.37 * p[2] - 0.25 * p[4];
  const y = 0.91 * p[1] - 0.32 * p[3] + 0.17 * p[4];
  const z = 0.58 * p[2] + 0.4 * p[3] + 0.36 * p[4];

  // Then rotate that 3D view around its vertical and horizontal axes.
  const cy = Math.cos(view.yaw);
  const sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch);
  const sp = Math.sin(view.pitch);
  const rotatedX = x * cy + z * sy;
  const yawZ = z * cy - x * sy;
  const rotatedY = y * cp - yawZ * sp;
  const depth = y * sp + yawZ * cp;

  // Finally, perspective 3D -> 2D.
  const perspective = 3.5 / (3.5 - 0.32 * depth);
  const scale = Math.min(width * 0.20, height * 0.25);
  return {
    x: width / 2 + rotatedX * perspective * scale,
    y: height / 2 + rotatedY * perspective * scale,
    depth,
  };
}
