import { useEffect, useRef, useState } from 'react';
import {
  edges as cubeEdges, fiveDPlanes, identity5D, projectVertex,
  rotate5D, vertices as cubeVertices,
} from './geometry.js';
import './viewer5d.css';

const DRAG_SENSITIVITY = 0.008;
const MAX_PITCH = Math.PI / 2 - 0.02;
const PLANE_BY_ID = Object.fromEntries(fiveDPlanes.map(plane => [plane.id, plane]));

// Optional shared controller lets two independent viewers rotate in lockstep.
// Internal controller remains the default, so hero and existing previews are unchanged.
export default function HypercubeCanvas({ playing = true, geometry = null, controller = null, showVectors = false, tint = 'cyan' }) {
  const canvasRef = useRef(null);
  const playingRef = useRef(playing);
  const localControllerRef = useRef(null);
  if (!localControllerRef.current) {
    localControllerRef.current = { orbit: { yaw: 0, pitch: 0 }, orientation: identity5D() };
  }
  const activeControllerRef = useRef(null);
  activeControllerRef.current = controller ?? localControllerRef.current;
  const dragRef = useRef(null);
  const [mode, setMode] = useState('orbit');
  const [horizontalPlane, setHorizontalPlane] = useState('0-3');
  const [verticalPlane, setVerticalPlane] = useState('1-4');
  playingRef.current = playing;

  function resetView() {
    activeControllerRef.current.orbit = { yaw: 0, pitch: 0 };
    activeControllerRef.current.orientation = identity5D();
  }

  function handleRotation(dx, dy) {
    if (mode === 'orbit') {
      activeControllerRef.current.orbit.yaw += dx * DRAG_SENSITIVITY;
      activeControllerRef.current.orbit.pitch = Math.max(
        -MAX_PITCH,
        Math.min(MAX_PITCH, activeControllerRef.current.orbit.pitch + dy * DRAG_SENSITIVITY),
      );
      return;
    }
    // In genuine 5D mode, horizontal and vertical motion rotate in separate
    // R^5 coordinate planes. The orthogonal matrix accumulates the rotations.
    const h = PLANE_BY_ID[horizontalPlane];
    const v = PLANE_BY_ID[verticalPlane];
    if (dx) rotate5D(activeControllerRef.current.orientation, h.a, h.b, dx * DRAG_SENSITIVITY);
    if (dy) rotate5D(activeControllerRef.current.orientation, v.a, v.b, dy * DRAG_SENSITIVITY);
  }

  function onPointerDown(event) {
    if (event.button !== 0 || dragRef.current !== null) return;
    event.preventDefault();
    dragRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.style.cursor = 'grabbing';
  }

  function onPointerMove(event) {
    const drag = dragRef.current;
    if (!drag || drag.id !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    handleRotation(dx, dy);
  }

  function onPointerEnd(event) {
    const drag = dragRef.current;
    if (!drag || drag.id !== event.pointerId) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    event.currentTarget.style.cursor = 'grab';
  }

  function onKeyDown(event) {
    const movements = { ArrowLeft: [-12, 0], ArrowRight: [12, 0], ArrowUp: [0, -12], ArrowDown: [0, 12] };
    const delta = movements[event.key];
    if (!delta) return;
    event.preventDefault();
    handleRotation(...delta);
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const vertices = geometry?.vertices ?? cubeVertices;
    const edges = geometry?.edges ?? cubeEdges;
    const context = canvas.getContext('2d');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let animationFrame = 0;
    let width = 1;
    let height = 1;
    let time = 0;
    let previousTime = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(rect.width, 1);
      height = Math.max(rect.height, 1);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const render = (timestamp) => {
      if (previousTime === 0) previousTime = timestamp;
      const delta = Math.min((timestamp - previousTime) / 1000, 0.1);
      previousTime = timestamp;
      if (playingRef.current && dragRef.current === null && !reducedMotion) time += delta;

      context.clearRect(0, 0, width, height);
      const view = activeControllerRef.current.orbit;
      const orientation = activeControllerRef.current.orientation;
      const positions = vertices.map(point => projectVertex(point, time, width, height, view, orientation));
      const sortedEdges = edges
        .map(([a, b]) => ({ a, b, depth: (positions[a].depth + positions[b].depth) / 2 }))
        .sort((left, right) => left.depth - right.depth);

      if (showVectors) {
        // In the dual panel each radius is a (scaled) primitive primal facet normal.
        const origin = projectVertex([0, 0, 0, 0, 0], time, width, height, view, orientation);
        for (const end of positions) {
          context.beginPath();
          context.moveTo(origin.x, origin.y);
          context.lineTo(end.x, end.y);
          context.strokeStyle = 'rgba(252, 191, 104, 0.55)';
          context.lineWidth = 1.15;
          context.stroke();
        }
      }

      for (const edge of sortedEdges) {
        const start = positions[edge.a];
        const end = positions[edge.b];
        const opacity = 0.23 + (edge.depth + 2.6) * 0.105;
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.strokeStyle = tint === 'violet'
          ? `rgba(181, 146, 255, ${Math.max(0.16, Math.min(opacity, 0.77))})`
          : `rgba(112, 220, 246, ${Math.max(0.16, Math.min(opacity, 0.77))})`;
        context.lineWidth = 1.1;
        context.stroke();
      }

      positions.forEach(({ x, y, depth }) => {
        const radius = 1.5 + (depth + 2) * 0.24;
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fillStyle = tint === 'violet' ? '#eee4ff' : '#dbfbff';
        context.fill();
      });
      animationFrame = window.requestAnimationFrame(render);
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    animationFrame = window.requestAnimationFrame(render);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
    };
  }, [geometry, showVectors, tint]);

  return (
    <div className="dimension-viewer">
      <div className="dimension-tools" aria-label="Rotation controls">
        <div className="dimension-mode" role="group" aria-label="Rotation mode">
          <button type="button" className={mode === 'orbit' ? 'active' : ''} aria-pressed={mode === 'orbit'} onClick={() => setMode('orbit')}>3D view</button>
          <button type="button" className={mode === '5d' ? 'active' : ''} aria-pressed={mode === '5d'} onClick={() => setMode('5d')}>True 5D</button>
        </div>
        {mode === '5d' && <div className="dimension-plane-controls">
          <label>↔
            <span className="sr-only-5d">Horizontal rotation plane</span>
            <select aria-label="Horizontal 5D rotation plane" value={horizontalPlane} onChange={e => setHorizontalPlane(e.target.value)}>
              {fiveDPlanes.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
          <label>↕
            <span className="sr-only-5d">Vertical rotation plane</span>
            <select aria-label="Vertical 5D rotation plane" value={verticalPlane} onChange={e => setVerticalPlane(e.target.value)}>
              {fiveDPlanes.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
        </div>}
        <button className="dimension-reset" type="button" onClick={resetView} title="Reset both the 3D camera and the 5D orientation">Reset</button>
      </div>
      <canvas
        ref={canvasRef}
        className="geometry-canvas"
        style={{ cursor: 'grab', touchAction: 'none' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        onLostPointerCapture={onPointerEnd}
        onKeyDown={onKeyDown}
        tabIndex={0}
        role="img"
        aria-label={mode === '5d'
          ? '5D polytope projection. Drag horizontally or vertically in the selected five-dimensional rotation planes, or use arrow keys.'
          : '5D polytope projection. Drag to orbit the 3D camera, or use arrow keys.'}
      />
    </div>
  );
}
