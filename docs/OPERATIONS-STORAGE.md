# Polytopes Atlas — Storage discipline and safe maintenance

Scope: this applies to the `web` and `api` containers of `apps/polytope5d/docker-compose.yml`. Traefik and other applications are managed elsewhere.

## Disk inventory (Ubuntu)

```bash
cd /opt/polytope5d
df -h
docker system df -v
docker compose ps
docker inspect -f '{{.Name}} {{.HostConfig.LogConfig.Type}} {{json .HostConfig.LogConfig.Config}}' polytope5d-web polytope5d-api
```

Look for `local` logging, `max-size=5m`, `max-file=2`. These settings take effect when Docker **recreates** each container after deployment; they do not retroactively truncate existing log files.

## Bounded logs

The Compose services explicitly set:

```yaml
logging:
  driver: "local"
  options:
    max-size: "5m"
    max-file: "2"
```

Log budget: about 10 MB per container, plus metadata. `docker compose logs --tail=50` remains available. Other Docker stacks and host logs are not covered by this limit.

## Persistent scientific data

The Docker volume `polytope5d-index` is mounted at `/data` in `api`. It stores `/data/polytopes.sqlite3` with the 3D catalogue and computed inclusion witnesses. Its contents are **not** deleted by rebuilding or recreating these containers. Preserve backups from `docs/DEPLOYMENT.md` before any storage maintenance.

4D live search retrieves bounded JSON pages from Hugging Face, performs exact geometric calculations in the browser, and keeps no full Parquet catalogue on disk. Browser memory is temporary.

## Conservative cleanup

Only after inspecting `docker system df -v` and taking a verified SQLite backup:

```bash
# Optionally remove dangling image layers not in use:
docker image prune

# Optionally remove old build cache (may slow the next build):
docker builder prune
```

Do NOT run `docker system prune --volumes`, `docker volume prune`, or remove `polytope5d-index` as routine cleanup. Do not clear the database to save space. Avoid deleting active JSON logs manually.

## Verify after a migration / deployment

```bash
cd /opt/polytope5d
docker compose config --quiet
docker compose ps
docker inspect -f '{{.Name}} {{.HostConfig.LogConfig.Type}} {{json .HostConfig.LogConfig.Config}}' polytope5d-web polytope5d-api
curl -fsS https://polytopes.mathematics.art/api/3d/stats
curl -fsS https://polytopes.mathematics.art/api/4d/live/status
```

Check `docker system df -v` weekly while performance and remote API traffic remain under observation.
