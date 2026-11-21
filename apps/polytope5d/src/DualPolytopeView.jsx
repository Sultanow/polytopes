import { useMemo, useRef, useState } from 'react';
import HypercubeCanvas from './HypercubeCanvas.jsx';
import { identity5D } from './geometry.js';
import './dualview.css';

function createController() {
  return { orbit: { yaw: 0, pitch: 0 }, orientation: identity5D() };
}

function copyController(source) {
  return {
    orbit: { ...source.orbit },
    orientation: source.orientation.map(row => [...row]),
  };
}

// Visual scaling ONLY: preserves the origin and never alters the stored lattice geometry.
function prepareGeometry(vertices, edges) {
  if (!Array.isArray(vertices) || vertices.length === 0) return null;
  const maxAbs = Math.max(1, ...vertices.flatMap(v => v.map(Math.abs)));
  const scale = 1.2 / maxAbs;
  return { vertices: vertices.map(v => v.map(x => x * scale)), edges: edges ?? [] };
}

export default function DualPolytopeView({ record }) {
  const [linked, setLinked] = useState(true);
  const [swapped, setSwapped] = useState(false);
  const [showNormals, setShowNormals] = useState(false);
  const [resetKey, setResetKey] = useState(0);
  const controllersRef = useRef(null);
  if (controllersRef.current === null) {
    controllersRef.current = {
      shared: createController(),
      left: createController(),
      right: createController(),
    };
  }
  const controllers = controllersRef.current;
  const primal = useMemo(() => prepareGeometry(record?.vertices_5d, record?.edges), [record]);
  const dual = useMemo(() => prepareGeometry(record?.dual_vertices_5d, record?.dual_edges), [record]);

  if (!record?.dual_available || !primal || !dual) return null;

  function changeLink(next) {
    if (next) {
      controllers.shared = copyController(controllers.left);
    } else {
      controllers.left = copyController(controllers.shared);
      controllers.right = copyController(controllers.shared);
    }
    setLinked(next);
  }

  function reset() {
    controllers.shared = createController();
    controllers.left = createController();
    controllers.right = createController();
    setResetKey(prev => prev + 1);
  }

  const left = swapped ? { name: 'Δ*', caption: 'Polar dual · facet normals of Δ',
    geometry: dual, count: record.dual_vertex_count, normalVectors: showNormals, tint: 'violet' }
    : { name: 'Δ', caption: 'Original polytope · exact stored lattice vertices',
      geometry: primal, count: record.vertex_count, normalVectors: false, tint: 'cyan' };
  const right = swapped ? { name: 'Δ', caption: 'Original polytope · exact stored lattice vertices',
    geometry: primal, count: record.vertex_count, normalVectors: false, tint: 'cyan' }
    : { name: 'Δ*', caption: 'Polar dual · facet normals of Δ',
      geometry: dual, count: record.dual_vertex_count, normalVectors: showNormals, tint: 'violet' };

  return (
    <section className="dualview" aria-label="Polar dual comparison">
      <div className="dualview-header">
        <div>
          <div className="dualview-kicker">03 / EXACT POLAR DUALITY · V0.5</div>
          <h3>One polytope. Two perspectives.</h3>
          <p>Exact 5D coordinates, projected into 2D. Drag either viewport to rotate.</p>
        </div>
        <div className="dualview-toolbar">
          <label className="dualview-toggle">
            <input type="checkbox" checked={linked} onChange={e => changeLink(e.target.checked)} />
            Link rotation
          </label>
          <label className="dualview-toggle">
            <input type="checkbox" checked={showNormals} onChange={e => setShowNormals(e.target.checked)} />
            Show normals
          </label>
          <button type="button" onClick={() => setSwapped(v => !v)}>Swap</button>
          <button type="button" onClick={reset}>Reset view</button>
        </div>
      </div>
      <div className="dualview-grid">
        {[left, right].map((panel, i) => (
          <div key={i} className="dualview-panel">
            <div className="dualview-panel-heading">
              <strong>{panel.name}</strong>
              <span>{panel.count} vertices · {panel.geometry.edges.length} exact edges</span>
            </div>
            <HypercubeCanvas
              key={`${i}-${resetKey}`}
              playing={false}
              geometry={panel.geometry}
              controller={linked ? controllers.shared : (i === 0 ? controllers.left : controllers.right)}
              showVectors={panel.normalVectors}
              tint={panel.tint}
            />
            <p>{panel.caption}</p>
          </div>
        ))}
      </div>
      <p className="dualview-footnote">
        Convention: Δ* = {'{'}u : ⟨u,v⟩ ≥ −1 for all v ∈ Δ{'}'}.
        {showNormals && ' Gold rays in the Δ* viewer visualize the primitive facet-normal vectors of Δ.'}
        {' '}The two drawings are scaled separately for readability; integer source coordinates are unchanged.
        {' '}Linked rotation synchronizes the 3D camera and the selected 5D orientation.
      </p>
    </section>
  );
}
