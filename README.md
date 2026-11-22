# Polytopes — research, algorithms, and interactive explorers

Research code and interactive tools for lattice polytopes, especially five-dimensional reflexive polytopes.

## Repository map

| Path | Purpose |
| --- | --- |
| [`fundamentals/`](fundamentals/) | Preserved research sources and experiments: C, Python, plots, data, docs, and original README. The code and its Git history are retained. |
| [`apps/polytope5d/`](apps/polytope5d/) | Polytope5D browser explorer (React/Vite, Docker, Traefik). |

The original README, including the previous genetic-algorithm results and examples, is preserved at [`fundamentals/README.md`](fundamentals/README.md).

## Polytope5D

A standalone web explorer with an animated 5D projection and an exact but **restricted** demonstrator for reflexive five-dimensional lattice simplices.

```bash
cd apps/polytope5d
npm install
npm test
npm run dev
```

For production, use `apps/polytope5d/deploy.ps1` or `docker compose up -d --build` inside that directory. It uses the existing external Docker network `n8n-net`.

## Research context

**Hatziiliou, Sultanow and Amir (2026)**, *Classification of Five-Dimensional Reflexive Polytopes Arising from Single Weight Systems*, AppliedMath 6(11), 194. https://doi.org/10.3390/appliedmath6110194

The article classifies Newton polytopes arising from the length-six **single weight system sector**. It does not enumerate combined weight systems or reflexive subpolytopes and does not claim an exhaustive classification of all 5D reflexive polytopes.

## Development roadmap

1. **v0.2** — 5D preview and bounded exact reflexive-simplex search (implemented, browser-only).
2. **v0.3** — JSON schema for canonical vertices, facets, invariants, and provenance; Three.js/WebGL-based manipulation of genuine 5D geometry projections.
3. **v0.4** — FastAPI service and a small index backed by a *documented and licensed* sample of the published catalogue.
4. **v1.0** — Scalable indexed lookup, queue-backed heavyweight jobs, cited provenance, and download/citation per result. Never load the entire multi-billion-record catalogue into a single browser or request.

**Migration note:** Paths inside legacy scripts may have assumed they were executed from the repository root. The move preserves files unchanged; adjust working directories when running older tools.

## Current status (2026-11-22)

- One React/Vite frontend for mathematics.art and the Polytopes Atlas (2D–5D).
- FastAPI + persistent SQLite index; 4,319 3D reflexive classes imported separately.
- v0.9 3D inclusion pilot: exact certificates for a partial set of relations, not the complete inclusion graph.

## Deployment and server migration

See [Deployment & Server Migration](docs/DEPLOYMENT.md) for installation, SQLite backup/restore, 3D catalogue import and inclusion jobs.
