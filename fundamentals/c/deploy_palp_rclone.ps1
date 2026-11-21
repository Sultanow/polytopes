$localPath  = "C:\Users\sulta\git\polytopes\c\palp-2.21"
$remotePath = "/root/palp/palp-2.21"

rclone sync $localPath "sultanowai:$remotePath" `
    --exclude ".git/**" `
    --exclude "*.o" `
    --exclude "*.exe" `
    --exclude "*.out" `
    -vv `
    --progress `
    --stats=1s

if ($LASTEXITCODE -ne 0) {
    Write-Host "Fehler aufgetreten. Taste drücken..."
    Read-Host
}