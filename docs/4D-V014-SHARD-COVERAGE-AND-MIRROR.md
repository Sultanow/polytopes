# Polytopes Atlas v0.14 — Alle 4D-Shards, Coverage und Batyrev-Spiegelsymmetrie

## Ziel und Datenquelle

- Quelle: https://huggingface.co/datasets/calabi-yau-data/polytopes-4d/tree/main
- Stand Dateiliste: 2026-11-22.
- **30 veröffentlichte Parquet-Dateien:** Eckenzahlen **5 bis 33 sowie 36**. Für 34/35 existiert in der überprüften Repository-Dateiliste kein eigener Shard.
- Gesamtklassifikation: 473.800.776 reflexive 4D-Gitterpolytope laut Datensatz und Kreuzer–Skarke.
- **Keine 15,8 GB als lokalen Datensatz spiegeln.** Die Anwendung fragt jeweils genau **eine** Remote-Parquet-Datei via DuckDB/httpfs ab.

## Neue API

- `GET /api/4d/remote/shards`: Statische, im Quellcode geprüfte Dateiliste. **Keine** externe Netzwerkabfrage, **kein** DuckDB-Scan und **keine** Zusicherung aktueller Live-Erreichbarkeit.
- `GET /api/4d/remote/status`: Sucht jetzt in allen 30 aufgelisteten Shards (ein Shard je HTTP-Anfrage). Sicherheitseinstellungen bleiben erhalten.
- `GET /api/4d/remote/count?vertex_count=8&h11=101`: Exakte Zählung **innerhalb** des ausgewählten Shards, sofern HTTP 200 mit `count_exact_for_shard: true` eintrifft.
- `GET /api/4d/remote/search?vertex_count=36&limit=1`: Einzelergebnisse in kleinen Shards (ausgewählte Shards 10–24 benötigen mindestens einen weiteren numerischen Filter).
- `GET /api/4d/remote/row/36/0`: Shard-relativer Direktzugriff. Die bereits vorhandenen Fingerprint-/Revisionsprüfungen aus v0.13 gelten weiter.

### Example requests

```bash
cd /opt/polytope5d

docker compose exec -T api \
  python -m unittest api.test_catalog4d_remote api.test_catalog4d_v013 api.test_catalog4d_v014 -v

docker compose exec -T api \
  python -m unittest api.test_catalog4d_remote_db -v

curl -fsS --max-time 15 \
  'https://polytopes.mathematics.art/api/4d/remote/shards'

curl -fsS --max-time 50 \
  'https://polytopes.mathematics.art/api/4d/remote/count?vertex_count=8&h11=101'

# Kleinen Tail-Shard testen; erst danach einen großen wie 16 oder 17.
curl -fsS --max-time 50 \
  'https://polytopes.mathematics.art/api/4d/remote/search?vertex_count=36&limit=1'
```

## Was die Coverage-UI bedeutet

Die Matrix zeigt alle 30 Shards. Für die **zuletzt abgeschickte Filterkombination** (ohne `vertex_count`) merkt sich der Browser je Shard:

1. `Not checked`: noch keine Abfrage ausgeführt.
2. `Page loaded`: eine Ergebnis-Seite erfolgreich gelesen, **kein** vollständiger Match-Count.
3. `Exact`: `COUNT(*)` erfolgreich beantwortet; exakte Zahl **in diesem Shard**.
4. `Failed / timed out`: keine aktuelle exakte Zählung; der Shard wird **nicht** als leer behandelt.

Die Summe aller 30 exakten Counts wird **erst** angezeigt, wenn alle 30 Shards mit **derselben Filterkombination** erfolgreich gezählt wurden. Bei mutabler HF-Revision `main` ist auch diese Summe nur eine zeitlich verteilte Bestandsaufnahme, keine atomare Datensatzmomentaufnahme. Eine festgepinnte `POLYTOPES_4D_REVISION` ist für reproduzierbare, umfassende Forschungsresultate vorzuziehen.

Die Browser-Matrix ist rein flüchtiger State: keine lokale Speicherpflicht, keine neue Datenbanktabelle, keine automatische 30-Shard-Schleife. Beim Neuladen beginnt die Matrix neu.

## Mathematische Darstellung: Hodge Mirror

Für die aus einem reflexiven 4D-Polytop zugeordnete Batyrev-Konstruktion einer Calabi–Yau-Dreifaltigkeit zeigt der Atlas die Hodge-Paarung

\[
(h^{1,1},h^{2,1})\quad\longleftrightarrow\quad(h^{2,1},h^{1,1}),
\qquad \chi=2(h^{1,1}-h^{2,1})\mapsto-\chi.
\]

Der Datensatz benutzt `h11` und `h12` (für `h^{1,2}=h^{2,1}`); die Darstellung prüft vor dem Anzeigen `chi == 2*(h11-h12)`. Bei inkonsistenten oder fehlenden Daten erscheint ein Warnhinweis. Die rechte Mirror-Zahl ist eine **theoretisch vorhergesagte Invariante**, nicht die Suche nach einem bereits identifizierten dualen Datensatz-Eintrag. Die vorhandene geometrische Dual-Canvas arbeitet unabhängig davon mit den Polytop-Eckkoordinaten.

Zur Einordnung: V. Batyrev, *Dual Polyhedra and Mirror Symmetry for Calabi–Yau Hypersurfaces in Toric Varieties* (1994); dazu https://www.mdpi.com/2073-8994/7/3/1633 .

## Speicher- und Lastbegrenzung

- Ein Remote-DuckDB-Prozess zeitgleich, weitere erhalten HTTP 429.
- Prozess-Abbruch nach 40 Sekunden mit HTTP 504.
- DuckDB-intern: 2 Threads, 256 MB Buffer Manager, maximal 32 MB temporärer Spill.
- **Keine garantierte harte Prozess-RSS- oder Netzwerk-Byte-Grenze.** Höhere Abfragevolumina großer Shards sind möglich; deswegen zunächst 8 und 36 testen und die größte Shards manuell, nur mit Filtern, prüfen.
- Es werden keine Parquet-Dateien dauerhaft gespeichert und keine neuen Docker-Volumes erzeugt. HTTPFS-Extension-Cache bleibt unter `/tmp` im API-Container.
- Exakte Trefferzahlen nur bei erfolgreichem `count` und `count_exact_for_shard: true`.

## Deployment und Rückfall

Der Patch baut auf v0.13 auf, ändert **nur** `catalog4d_remote.py`, `test_catalog4d_remote.py`, die neuen v0.14-Tests, `Dimension4DLiveExplorer.jsx`, `catalog4d-live.css`, die Mirror- und Coverage-Helfer und dieses Dokument. Keine Änderungen an 2D/3D/5D, Landingpage, SQLite, Compose, Traefik oder SSL.

```powershell
cd 'C:\Users\sulta\git\polytopes'
& 'C:\Users\sulta\Downloads\polytope-atlas-v014-all-shards-coverage-mirror\apply-v014.ps1' -CheckOnly
& 'C:\Users\sulta\Downloads\polytope-atlas-v014-all-shards-coverage-mirror\apply-v014.ps1'

git diff --check
git add apps/polytope5d/api apps/polytope5d/src/Dimension4DLiveExplorer.jsx apps/polytope5d/src/catalog4d-live.css apps/polytope5d/src/hodgeMirror4d.mjs apps/polytope5d/src/hodgeMirror4d.test.mjs apps/polytope5d/src/coverage4d.mjs apps/polytope5d/src/coverage4d.test.mjs docs/4D-V014-SHARD-COVERAGE-AND-MIRROR.md
git commit -m 'Expand 4D shard coverage and show Hodge mirror pairs'
git push
& '.\apps\polytope5d\deploy.ps1'
```

Bei Fehlverhalten den letzten Commit revertieren und erneut deployen; SQLite und die existierenden 4D-Links bleiben unverändert.
