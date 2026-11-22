# mathematics.art platform shell — v0.7a

One React/Vite build, one Nginx container and one existing FastAPI service for both domain names.

## URLs

| URL | Result |
| --- | --- |
| https://mathematics.art/ | New landing page, background `#727272` |
| https://mathematics.art/polytopes | 308 -> https://polytopes.mathematics.art/ |
| https://mathematics.art/polytopes/3d | 308 -> https://polytopes.mathematics.art/3d |
| https://polytopes.mathematics.art/ | Existing v0.6 atlas homepage |
| https://polytopes.mathematics.art/2d through /5d | Existing dimension pages |
| https://polytopes.mathematics.art/compute | Existing concept page |
| https://polytope5d.mathematics.art/ | Original 5D app |
| https://mathematics.art/polytope5d/ | Original 5D app (if apex DNS available) |

**DNS:** `mathematics.art` needs its own A/AAAA record pointing to the server. `*.mathematics.art` does not cover the apex. Existing HTTPS services on `mathematics.art` with competing router rules should be checked before deployment. Traefik rules here match only `/`, `/polytopes`, and `/polytopes/…` and retain the existing `/polytope5d` router.

## Team photos

`public/team/eldar.svg`, `malik.svg`, and `andreas.svg` are intentionally abstract placeholders, NOT photos of real people.

When ready, copy your photos into `public/team/` as `eldar.jpg`, `malik.jpg`, `andreas.jpg`, and update only these three `image` strings in `src/MathematicsHome.jsx`:

```js
{ name: 'Eldar', image: '/team/eldar.jpg', ... }
{ name: 'Malik', image: '/team/malik.jpg', ... }
{ name: 'Andreas', image: '/team/andreas.jpg', ... }
```

Finally remove or update the temporary `math-photo-note` label in `MathematicsHome.jsx`. Keep images appropriately sized/compressed; do not commit image files for which you have no permission.

## Technical boundaries

- Existing API, database volume, Dockerfile, build process and deployment script stay unchanged.
- No new npm or Python dependencies.
- `DimensionPortal.jsx` chooses the mathematics landing page based on hostname, preserving v0.6 atlas routing.
- The root `README.md`, `fundamentals` folder and repository structure remain unchanged.
- The 3D full classification module is planned for v0.7b and is not part of this patch.
