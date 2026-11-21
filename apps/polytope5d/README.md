# Polytope5D – Mouse rotation patch (v0.3)

Nur 2 Dateien ersetzen (in deinem lokalen Repo):

- `apps/polytope5d/src/HypercubeCanvas.jsx`
- `apps/polytope5d/src/geometry.js`

Danach `cd apps/polytope5d` und `./deploy.ps1` aufrufen.

Funktionen:
- Linke Maustaste gedrückt halten und ziehen: Blick drehen (Yaw/Pitch).
- Touch / Stift: ebenfalls drehen.
- Auto-Rotation pausiert während des Ziehens.
- Vorhandener Pause/Resume-Knopf funktioniert unverändert.
- Sowohl im Hero-Fenster als auch in der Katalog-Detailvorschau.

Die Benutzerinteraktion verändert nur die Kameraausrichtung der projizierten 3D-Ansicht, nicht die exakten 5D-Koordinaten des Polytops.
