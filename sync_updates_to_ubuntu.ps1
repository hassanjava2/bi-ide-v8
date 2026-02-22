param(
    [string]$UbuntuUser = "bi",
    [string]$UbuntuHost = "192.168.68.125",
    [string]$RemoteProjectPath = "~/bi-ide-v8"
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

$filesToSync = @(
    "start_training_ubuntu.sh",
    "RTX4090_SETUP.md",
    "rtx4090_server.py"
)

Write-Host ("Syncing updates to {0}@{1}:{2}" -f $UbuntuUser, $UbuntuHost, $RemoteProjectPath) -ForegroundColor Cyan

foreach ($file in $filesToSync) {
    if (-not (Test-Path $file)) {
        throw "Missing file: $file"
    }

    $localPath = Join-Path $projectRoot $file
    $remotePath = "$UbuntuUser@$UbuntuHost`:$RemoteProjectPath/$file"

    Write-Host "Uploading $file ..." -ForegroundColor Yellow
    scp "$localPath" "$remotePath"
}

Write-Host "Applying executable permission for start script on Ubuntu..." -ForegroundColor Yellow
ssh "$UbuntuUser@$UbuntuHost" "chmod +x $RemoteProjectPath/start_training_ubuntu.sh && ls -l $RemoteProjectPath/start_training_ubuntu.sh"

Write-Host "Done. Files synced successfully." -ForegroundColor Green
