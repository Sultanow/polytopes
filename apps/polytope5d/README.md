# Polytope5D — True 5D Rotation (v0.4)

**Manual upgrade** for the running `Sultanow/polytopes` project, following the v0.3 API and mouse-orbit updates.

Copy or replace ONLY these 3 files in `polytopes/apps/polytope5d/src/`:

1. `src/HypercubeCanvas.jsx` — replaces the mouse rotation component
2. `src/geometry.js` — replaces the 5D projection utilities
3. `src/viewer5d.css` — new stylesheet

No change required to `App.jsx`, `CatalogueSearch.jsx`, the backend, or Docker.

## Usage

- **3D view** (default): click-drag to orbit the 3D view, just like before.
- **True 5D**: click-drag to apply genuine orthogonal 5D rotations.
  - Horizontal drag uses the plane chosen under **↔** (initially x1–x4).
  - Vertical drag uses the plane chosen under **↕** (initially x2–x5).
  - Each dropdown offers all 10 pairs of the five coordinate axes.
- **Reset**: restores the initial 3D camera and 5D orientation.
- Touch, stylus, and keyboard arrow keys are supported (focus the canvas first).
- Existing Pause/Resume continues to control the background auto-rotation on the hero canvas. For precise inspections, press **Pause** first.

The original integer coordinates are **never modified**. The displayed coordinates are transformed by an orthogonal 5×5 matrix, followed by the existing animation and a 5D→3D→2D projection. A display-space rotation does **not** mean that the lattice polytope has been changed into another lattice polytope.

The same `HypercubeCanvas` component is shared by the hero and the catalogue preview. No new dependencies are added.

## Deploy

```powershell
cd apps/polytope5d
.\deploy.ps1
```

If you want to run the optional math tests, also copy `tests/rotation5d.test.mjs` to your project:

```powershell
node --test tests/rotation5d.test.mjs
```

## Scope / validation

Validated here: the 10 plane choices, 5D Givens-rotation arithmetic, orthogonality, conservation of dot products, immutable source coordinates, 32/80 tesseract counts, and JSX syntax using TypeScript transpilation. The full React/Docker build and browser pointer interactions still need to be checked on your deployed application.
