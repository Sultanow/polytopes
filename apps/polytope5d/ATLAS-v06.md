# Polytopes Atlas v0.6 — URLs + exact 2D classification + 3D/4D previews

The multi-dimensional atlas adds these routes without replacing the existing 5D explorer:

- `https://polytopes.mathematics.art/`: dimension overview.
- `/2d`: all 16 2D reflexive lattice polygon classes, exact integer vertices, 22 verified direct inclusion cover relations, and a composed nested chain.
- `/3d`: interactive 3D cube demonstration; complete 4319 polytope import is not part of this patch.
- `/4d`: interactive projection of a real 4D simplex from the Hugging Face dataset viewer, with source attribution. Complete 4D data is not imported.
- `/5d`: current v0.5 viewer unchanged.
- `/compute`: concept page only, no worker service.

The existing `https://polytope5d.mathematics.art/` continues to serve the original 5D view. The existing `https://mathematics.art/polytope5d/` also remains configured.

## Data and accuracy

**2D:** The representatives were independently constructed by a finite bounded search in `[-3,3]^2`, tested with primitive facet equations, deduplicated under integer unimodular transformations, and cross-checked to the known classification count 16. Their stored integer vertices are real and exactly reflexive. Each displayed inclusion comes with an explicitly verified GL(2,Z) matrix; the graph was found with transformation coefficients within `[-8,8]`. It has 22 direct edges under the *computed* relation. This bounded search does not by itself prove there are no embeddings outside that bound.

Regenerate the data with `python scripts/generate_reflexive_2d.py`; run `python scripts/test_reflexive_2d.py` for exact checks. There are no third-party dependencies for these scripts.

**3D:** Source: https://github.com/sagemath/data-polytopes . SageMath's database package carries a caveat about unclear rights for redistributing the original polygon/polyhedron contents. No third-party 3D catalogue is bundled in this patch. See `DATA_LICENSES.md` in that project and clarify redistribution terms before deploying a data mirror.

**4D:** Source: https://huggingface.co/datasets/calabi-yau-data/polytopes-4d (Kreuzer–Skarke catalogue compiled as Parquet; CC BY-SA 4.0). The interactive sample uses the first row that the dataset viewer displays: `[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,2,3,5],[-2,-3,-4,-5]]`. The entire dataset (15.8 GB; ~474M records) is **not** downloaded into the frontend or our server at this stage. Production search should import selected Parquet shards offline and build an indexed, paginated backend.

**5D:** The v0.5 FastAPI, SQLite and dual viewer remain unchanged.

## Deployment

Patch from the root of your clean, committed `polytopes` repository, then commit and run the unchanged `apps/polytope5d/deploy.ps1`.

Ensure `*.mathematics.art` wildcard DNS resolves to the Traefik server. A certificate for `polytopes.mathematics.art` is issued by the existing `letsencrypt` resolver once the Traefik router is active.

No new ports, environment variables, Node dependencies, Python backend requirements or databases are introduced.

## Architecture after this patch

We intentionally reuse one React/Vite build and one Traefik/Nginx service for all public dimensional routes. This keeps deployment trivial and allows independent backend APIs later: `/api/2d/...`, `/api/3d/...`, `/api/4d/...` and `/api/5d/...`. At present only the existing v0.5 `/api/...` exists; this patch does **not** implement 2D/3D/4D API endpoints.

`vite.config.js` retains `base: './'` for compatibility with the old `/polytope5d/` path. Therefore Nginx normalizes trailing-slash requests to `/2d/`, `/3d/`, `/4d/`, `/5d/`, `/compute/` back to slashless paths so the compiled JS/CSS assets resolve correctly.

## Follow-on tasks

1. Licensed 3D data ingestion plus exact minimal/maximal inclusion graph.
2. Offline Parquet ingestion of a small 4D shard into a queryable dimension-aware index.
3. 5D research catalogue indexing and reproducible provenance/canonical IDs.
4. A volunteer compute protocol with bounded tasks, authenticated submissions, checksums, duplicate computation, independent certificate verification, and rate limits.
