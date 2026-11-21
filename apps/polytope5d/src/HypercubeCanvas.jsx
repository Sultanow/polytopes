import { useEffect, useRef } from 'react';
import { edges as cubeEdges, projectVertex, vertices as cubeVertices } from './geometry.js';

const DRAG_SENSITIVITY = 0.008; // radians per pointer pixel
const MAX_PITCH = Math.PI / 2 - 0.02;

export default function HypercubeCanvas({ playing, geometry = null }) {
  const canvasRef = useRef(null);
  const playingRef = useRef(playing);
  const orbitRef = useRef({ yaw: 0, pitch: 0 });
  const dragRef = useRef(null);
  playingRef.current = playing;

  function onPointerDown(event) {
    // Left mouse button or primary touch/pen only.
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
    orbitRef.current.yaw += dx * DRAG_SENSITIVITY;
    orbitRef.current.pitch = Math.max(
      -MAX_PITCH,
      Math.min(MAX_PITCH, orbitRef.current.pitch + dy * DRAG_SENSITIVITY),
    );
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
      // Auto rotation pauses while the user is dragging.
      if (playingRef.current && dragRef.current === null && !reducedMotion) time += delta;

      context.clearRect(0, 0, width, height);
      const view = orbitRef.current;
      const positions = vertices.map(point => projectVertex(point, time, width, height, view));
      const sortedEdges = edges
        .map(([a, b]) => ({ a, b, depth: (positions[a].depth + positions[b].depth) / 2 }))
        .sort((left, right) => left.depth - right.depth);

      for (const edge of sortedEdges) {
        const start = positions[edge.a];
        const end = positions[edge.b];
        const opacity = 0.23 + (edge.depth + 2.6) * 0.105;
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.strokeStyle = `rgba(112, 220, 246, ${Math.max(0.16, Math.min(opacity, 0.77))})`;
        context.lineWidth = 1.1;
        context.stroke();
      }

      positions.forEach(({ x, y, depth }) => {
        const radius = 1.5 + (depth + 2) * 0.24;
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fillStyle = '#dbfbff';
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
  }, [geometry]);

  return (
    <canvas
      ref={canvasRef}
      className="geometry-canvas"
      style={{ cursor: 'grab', touchAction: 'none' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onLostPointerCapture={onPointerEnd}
      role="img"
      aria-label="Interactive 2D projection of a five-dimensional polytope. Drag to rotate the 3D view."
    />
  );
}
