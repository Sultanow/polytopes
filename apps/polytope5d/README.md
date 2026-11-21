# Polytope5D · Interactive research preview v0.2

A standalone React/Vite frontend built into a static Nginx container, deployed with Traefik. Compatible with the existing external `n8n-net` on the server.

## Scope

- Rotating projection of a genuine 5D hypercube: 32 vertices, 80 edges. An optional **Artwork** tab uses `public/hero-polytope.png` as a purely conceptual illustration, explicitly distinguished from exact geometry.
- Search and visualization of a bounded **simplex family** with weight systems `(1, q1, q2, q3, q4, q5)` where `qi` are positive integers.
- The simplex `Δ = conv((-q1, ...,-q5), e1, ..., e5)` contains the origin in its interior and is reflexive exactly if `qi | (1 + sum(qi))` for each i. The exact facet normals and lattice vertices are inspectable in the UI.
- For the bounded search, `q1 <= ... <= q5 <= 12`; the nondecreasing convention removes simple weight permutations, **not** all GL(5,Z)-equivalences. No complete catalogue is searched.
- The published paper covers Newton polytopes from all length-six *single weight systems*, a broader class than this particular simplex model, but **not** all reflexive 5D polytopes.

## Local development

```bash
cd apps/polytope5d
npm install
npm test
npm run dev
```

## Deploy

From a Windows PowerShell at `apps/polytope5d`:

```powershell
.\deploy.ps1
```

The script uses the SSH key in `~/.ssh/id_ed25519_20260412` and uploads only the Polytope5D app to `/opt/polytope5d`. It does **not** run `docker compose down` on other projects.

Container setup: `node:22-alpine` builds static assets, `nginx:1.28-alpine` serves port 8080 (exposed only on Docker's `n8n-net`). Traefik provides HTTPS on the `websecure` entrypoint, using the existing `letsencrypt` cert resolver.

Supported URLs:
- `https://polytope5d.mathematics.art/`
- `https://mathematics.art/polytope5d/` (requires separate apex-domain DNS and HTTPS routing)

If you want a deterministic npm build, run `npm install` locally and commit the generated `package-lock.json`; the Dockerfile will use `npm ci` when present.

## Future search-engine architecture

The real multi-billion-record catalogue warrants a separate indexed backend. Suggested split:

- `apps/polytope5d/` — frontend, projection and UX, types, lightweight search controls.
- `services/polytope5d-api/` (later) — FastAPI for exact catalogue IDs, weight-vector lookup, GL-normal forms, vertex/facet counts, dual/Hodge invariants, filters.
- Dedicated read-only index or catalogue shard storage (not included). Precompute compact metadata and provenance; do not expose multi-TB raw archives in HTTP requests.
- Optional queue/worker for computational searches; complexity and resource limits with explicit cancellation.

## Research

Hatziiliou, Sultanow, Amir: [Classification of Five-Dimensional Reflexive Polytopes Arising from Single Weight Systems](https://www.mdpi.com/2673-9909/6/11/194), *AppliedMath* 6(11), 194 (2026).
