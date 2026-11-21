# Auf Windows (PowerShell 7 oder Windows PowerShell 5.1) ausführen:
#   .\deploy.ps1
# Optional: .\deploy.ps1 -Server root@37.27.119.115 -KeyFile C:\Pfad\zu\id_ed25519
param(
    [string]$Server = "root@37.27.119.115",
    [string]$RemoteBase = "/opt/polytope5d",
    [string]$KeyFile = (Join-Path $HOME ".ssh\id_ed25519_20260412")
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sshOptions = @("-i", $KeyFile, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes")

if (-not (Test-Path $KeyFile)) {
    throw "SSH-Key nicht gefunden: $KeyFile"
}

function Check-Exit([string]$Message) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Message (Exitcode: $LASTEXITCODE)"
    }
}

Write-Host "[1/5] Pruefe Docker-Netzwerk und erzeuge Projektverzeichnis..." -ForegroundColor Cyan
& ssh @sshOptions $Server "docker network inspect n8n-net >/dev/null 2>&1 && mkdir -p '$RemoteBase'"
Check-Exit "n8n-net fehlt auf dem Server oder SSH-Verbindung fehlgeschlagen"

Write-Host "[2/5] Uebertrage Projektdateien..." -ForegroundColor Cyan
$files = @("Dockerfile", "docker-compose.yml", "nginx.conf", "package.json", "vite.config.js", "index.html", ".dockerignore")
foreach ($file in $files) {
    & scp @sshOptions (Join-Path $ProjectRoot $file) "${Server}:${RemoteBase}/"
    Check-Exit "Upload fehlgeschlagen: $file"
}

# Altes Projektmaterial NUR im eigenen Verzeichnis entfernen, nicht bei KMDL.
& ssh @sshOptions $Server "rm -rf '$RemoteBase/src' '$RemoteBase/public' '$RemoteBase/api'"
Check-Exit "Bereinigung der Projektquellen fehlgeschlagen"
foreach ($folder in @("src", "public", "api")) {
    & scp @sshOptions -r (Join-Path $ProjectRoot $folder) "${Server}:${RemoteBase}/"
    Check-Exit "Upload fehlgeschlagen: $folder"
}

Write-Host "[3/5] Baue und starte ausschliesslich Polytope5D und seine API..." -ForegroundColor Cyan
& ssh @sshOptions $Server "cd '$RemoteBase' && docker compose up -d --build --wait"
Check-Exit "Docker-Deployment fehlgeschlagen"

Write-Host "[4/5] Containerstatus:" -ForegroundColor Cyan
& ssh @sshOptions $Server "cd '$RemoteBase' && docker compose ps"
Check-Exit "Statusabfrage fehlgeschlagen"

Write-Host "[5/5] Gesundheitscheck:" -ForegroundColor Cyan
& ssh @sshOptions $Server "cd '$RemoteBase' && docker compose exec -T web wget -qO- http://127.0.0.1:8080/healthz"
Check-Exit "Gesundheitscheck fehlgeschlagen"

Write-Host "Fertig! https://polytope5d.mathematics.art/" -ForegroundColor Green
Write-Host "Alternativ (falls Apex-DNS existiert): https://mathematics.art/polytope5d/"
