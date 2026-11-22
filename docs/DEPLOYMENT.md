# Polytopes / mathematics.art — Deployment & Migration Runbook

Stand: 2026-11-22 (Atlas v0.9 / 3D-Inklusionspilot)

Dieses Dokument beschreibt das **Wiederaufsetzen auf einem neuen Ubuntu-Server** und die dafür nötigen **einmaligen Datenoperationen**. Die eigentliche Produktivkonfiguration im Repository (`apps/polytope5d/docker-compose.yml` und `deploy.ps1`) hat Vorrang vor historischen Patch-READMEs.

## 0. Architektur und wichtige Daten

| Bestandteil | Aktueller Aufbau |
| --- | --- |
| Git | `https://github.com/Sultanow/polytopes`, App unter `apps/polytope5d/` |
| Frontend | React/Vite → Nginx, Compose-Service `web`, Container `polytope5d-web` |
| Backend | FastAPI, Compose-Service `api`, Container `polytope5d-api` |
| Datenbank | SQLite `POLYTOPE_DB=/data/polytopes.sqlite3` |
| Persistenz | Compose-Named-Volume (derzeit `polytope5d-index` in der Compose-Datei; Docker-Projektpräfix beachten) auf `/data` der API |
| Netzwerk | Externes Docker-Netz `n8n-net` für Traefik, internes App-Netz für Web/API |
| Serverpfad | `/opt/polytope5d` |
| Live | `https://mathematics.art/`, `https://polytopes.mathematics.art/`, `/2d`, `/3d`, `/4d`, `/5d` |
| Weitere Subdomain | `polytope5d.mathematics.art` (Kompatibilität) |

Der SQLite-Index enthält neben 3D-Katalog und 3D-Inklusionszertifikaten auch die übrigen lokalen App-Daten. **Ein Git-Clone stellt dieses Daten-Volume nicht wieder her.** Keine Datenbank und keine Traefik-ACME-Dateien im öffentlichen Repository committen.

Die Berechnung des v0.9-Inklusionspiloten ist **nicht vollständig**: Sie überprüft nur 48 signierte Koordinatenpermutationen und liefert zertifizierte *Teil*-Relationen, keinen vollständigen Cover- oder Hasse-Graphen.

## 1. Voraussetzungen auf dem neuen Ubuntu-Server

1. Docker Engine und Docker Compose Plugin (`docker compose`) installieren und prüfen: `docker version`, `docker compose version`.
2. Einen funktionierenden **Traefik**-Stack mit EntryPoints `web` (Port 80), `websecure` (Port 443) und ACME-Resolver **`letsencrypt`** bereitstellen. In der neuen Umgebung müssen diese Namen zu den Labels im Compose passen.
3. Das **externe** Docker-Netz `n8n-net` muss existieren und von Traefik verwendet werden. Prüfen: `docker network inspect n8n-net`. Nicht blind ein zweites Traefik-Netz anlegen oder mehrere Traefik-Instanzen auf 80/443 betreiben.
4. SSH-Zugriff mit Schlüssel auf den neuen Server und Schreibberechtigung für `/opt/polytope5d`; Ports 80/443 erreichbar.
5. Vor dem produktiven Umschalten die DNS-A-Records für `mathematics.art` und `polytopes.mathematics.art` (ggf. auch die Alt-Subdomain) planen. AAAA nur setzen, wenn die IPv6-Adresse korrekt konfiguriert ist. Zertifikate für neue Hosts werden vom ACME-Resolver angefordert; nicht voraussetzen, dass bisherige Zertifikate automatisch mitziehen.

**Wichtig:** Traefik selbst und seine ACME-Konfiguration gehören zum bestehenden Infrastruktur-Stack und werden nicht durch das Polytopes-Repository automatisch mitdeployt. Änderungen dort separat dokumentieren. Keine Passwörter, privaten SSH-Schlüssel oder `acme.json` öffentlich ablegen.

## 2. Migration: konsistentes Datenbank-Backup auf dem ALTEN Server

Auf dem bisherigen Server (Ubuntu, als Benutzer mit Docker-Zugriff):

```bash
cd /opt/polytope5d
mkdir -p /root/polytope-backups
chmod 700 /root/polytope-backups

# Datenbankstand sicher online via SQLite-Backup-API sichern.
# Das vermeidet eine inkonsistente Kopie bei gleichzeitigen Schreibvorgängen.
docker compose exec -T api python - <<'PY'
import os, sqlite3
source = os.environ.get('POLYTOPE_DB', '/data/polytopes.sqlite3')
target = '/data/polytopes-transfer-backup.sqlite3'
with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as src:
    with sqlite3.connect(target) as dst:
        src.backup(dst)
with sqlite3.connect(target) as check:
    result = check.execute('PRAGMA integrity_check').fetchone()[0]
    if result != 'ok':
        raise SystemExit(f'BACKUP INVALID: {result}')
    print('SQLite backup integrity:', result)
PY

# Aus dem bestehenden API-Container auf den Host kopieren.
docker cp polytope5d-api:/data/polytopes-transfer-backup.sqlite3 \
  /root/polytope-backups/polytopes.sqlite3
sha256sum /root/polytope-backups/polytopes.sqlite3 \
  > /root/polytope-backups/polytopes.sqlite3.sha256

# Temporäre Kopie im Volume entfernen; Originaldatenbank bleibt unberührt.
docker compose exec -T api rm /data/polytopes-transfer-backup.sqlite3
```

Zusätzlich als **nicht geheime Diagnoseinformationen** sichern:

```bash
curl -fsS https://polytopes.mathematics.art/api/3d/stats \
  > /root/polytope-backups/catalogue3d-stats.json
curl -fsS https://polytopes.mathematics.art/api/3d/inclusions/stats \
  > /root/polytope-backups/inclusions3d-stats.json
cd /opt/polytope5d && docker compose ps
```

Backup geschützt auf ein anderes System übertragen, z. B. vom alten Server aus (NEUE_IP ersetzen):

```bash
scp /root/polytope-backups/polytopes.sqlite3* root@NEUE_IP:/root/
```

**Nicht** nur `docker cp` der *laufenden* Datei `polytopes.sqlite3` machen: Bei offenen SQLite-Transaktionen/WAL kann das inkonsistent sein. Vor dem endgültigen DNS-Umschalten einen letzten konsistenten Backup-Snapshot erstellen, wenn inzwischen weitere Inklusionsjobs gelaufen sind. Backup außerhalb des Servers aufbewahren.

## 3. Anwendung auf dem NEUEN Server deployen

Auf Windows im lokalen Git-Checkout (gegebenenfalls zuerst `git pull`):

```powershell
cd "C:\Users\sulta\git\polytopes"
git pull

# IP / SSH-Ziel anpassen; der App-Code wird unter /opt/polytope5d deployt.
& ".\apps\polytope5d\deploy.ps1" -Server "root@NEUE_IP" -RemoteBase "/opt/polytope5d"
```

Die `deploy.ps1`-Datei erwartet standardmäßig einen konfigurierten SSH-Key; bei Bedarf `-KeyFile "C:\Pfad\zum\Schluessel"` ergänzen. Das Skript kopiert `src/`, `public/`, `api/` sowie die Konfigurationen und führt `docker compose up -d --build --wait` aus. **Vorher** muss `n8n-net` existieren.

Auf dem neuen Ubuntu-Server prüfen:

```bash
cd /opt/polytope5d
docker compose config --services
docker compose ps
```

### 3a. Bestehenden SQLite-Index auf neuem Server wiederherstellen (empfohlen)

**Nur auf dem neuen Server ausführen**, nachdem die Datenbanksicherung dorthin kopiert wurde. Der API-Service wird während des Austauschs angehalten; der Web-Service kann kurzzeitig API-Fehler melden.

```bash
cd /opt/polytope5d
docker compose stop api

# In einem einmaligen API-Container mit Root-Rechten die Datei im
# gemounteten Daten-Volume ersetzen und die Eigentümerrechte korrigieren.
# Passe /root/polytopes.sqlite3 an den tatsächlichen Backup-Pfad an.
docker compose run --rm --no-deps --user root \
  -v /root/polytopes.sqlite3:/tmp/restore.sqlite3:ro \
  api sh -c 'rm -f /data/polytopes.sqlite3-wal /data/polytopes.sqlite3-shm && cp /tmp/restore.sqlite3 /data/polytopes.sqlite3 && chown 10001:10001 /data/polytopes.sqlite3 && chmod 600 /data/polytopes.sqlite3'

docker compose up -d api

docker compose exec -T api python - <<'PY'
import os, sqlite3
p = os.environ.get('POLYTOPE_DB', '/data/polytopes.sqlite3')
with sqlite3.connect(p) as db:
    print('integrity:', db.execute('PRAGMA integrity_check').fetchone()[0])
    print('3D classes:', db.execute('SELECT COUNT(*) FROM catalogue3d').fetchone()[0])
    print('certified 3D relations:', db.execute('SELECT COUNT(*) FROM catalogue3d_inclusion').fetchone()[0])
PY
```

**Achtung:** Der Restore überschreibt den Index auf dem *neuen* Server. Nie auf dem alten Produktivserver ausführen. Der Container-UID `10001` stammt aus dem aktuellen API-Dockerfile; bei Änderungen an `USER`/`UID` prüfen und anpassen.

### 3b. Alternative: Index neu erstellen (ohne Datenbank-Backup)

Nur wenn kein vorhandenes Backup übernommen werden soll. Die publizierten 3D-Klassen werden aus dem beim API-Docker-Build installierten SageMath-Datenpaket eingelesen (kein komplettes SageMath zur Laufzeit erforderlich):

```bash
cd /opt/polytope5d

docker compose exec -T api python -m unittest api.test_catalog3d -v
docker compose exec -T api python -m api.catalog3d_import

# Erwartet: OK: 4319 validated 3D classes, sha256=...
docker compose exec -T api python -m unittest api.test_catalog3d_inclusion -v

# v0.9-Pilot: erste 20 Quellen gegen 4319 Ziele prüfen
docker compose exec -T api python -m api.catalog3d_inclusion \
  --source-start 0 --source-end 20 \
  --target-start 0 --target-end 4319
```

Weitere Batches wie `--source-start 20 --source-end 40` sind möglich. **Ein neuer Import kann vorhandene Inklusionsresultate ungültig machen**, weil diese per SHA-256 an den importierten Katalog gebunden sind. Deshalb bevorzugt den vollständigen Index *inklusive* Inklusionen per Backup migrieren.

## 4. Umschalten, Smoke-Tests und Rückfallplan

Auf dem neuen Server oder extern prüfen (nach DNS-Umschaltung):

```bash
curl -I https://mathematics.art/
curl -I https://polytopes.mathematics.art/
curl -fsS https://polytopes.mathematics.art/api/3d/stats
curl -fsS https://polytopes.mathematics.art/api/3d/inclusions/stats
```

Zusätzlich im Browser die Seiten `/2d`, `/3d`, `/4d`, `/5d` testen, inklusive 3D-Viewer, Dualansicht und Team-Bildern. Bei Zertifikatsproblemen mit `openssl s_client -connect NEUE_IP:443 -servername polytopes.mathematics.art` DNS/TLS einzeln prüfen.

**Rollback:** Alte Maschine während der Übergangsphase unverändert erreichbar lassen. Wenn neue Infrastruktur fehlschlägt, DNS-A-Record auf bisherige IP zurückstellen und die Ursachen auf dem neuen Server prüfen. **Niemals** bei der Fehlersuche `docker compose down -v` auf dem bestehenden Produktivsystem ausführen, da dies das Daten-Volume löschen kann.

## 5. Betrieb: was regelmäßig zu sichern ist

- **Zwingend:** SQLite-Index `/data/polytopes.sqlite3` über die konsistente Backup-Methode aus Abschnitt 2 (enthält auch die Berechnungsergebnisse).
- **Reproduzierbarkeit:** Git-Commit oder Tag der eingesetzten Version notieren, sowie Docker/Compose-Version, Quelle und Prüfsumme der importierten 3D-Daten.
- **Infrastruktur separat:** Traefik-Konfiguration (ohne Veröffentlichung sensibler Werte), DNS, Reverse-Proxy-Router und ACME-Resolver sichern.
- **Secrets:** SSH-Schlüssel und etwaige geheime `.env`-Dateien gehören *nicht* ins Git-Repo.
- **Datenrechte:** Bedingungen zur Weitergabe des SageMath-3D-Klassifikationsbestands vor einer vollständigen öffentlichen Spiegelung klären.

Im Gegensatz zu den ursprünglichen Patch-READMEs ist dieses Runbook die zentrale Anleitung für eine **frische Installation, Wiederherstellung und Servermigration**. Bei künftigem Wechsel von SQLite zu PostgreSQL muss insbesondere der Backup-/Restore-Abschnitt entsprechend aktualisiert werden.
