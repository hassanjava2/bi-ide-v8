param(
    [string]$ServerUrl = $env:ORCH_SERVER,
    [string]$Token = $env:ORCH_TOKEN,
    [string]$WorkerName = $env:WORKER_NAME,
    [string]$Labels = "desktop,autonomous,builder",
    [int]$PollSec = 5
)

if (-not $ServerUrl -or $ServerUrl.Trim() -eq "") { $ServerUrl = "http://localhost:8000" }
if (-not $WorkerName -or $WorkerName.Trim() -eq "") { $WorkerName = "desktop-$env:COMPUTERNAME" }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$agentPath = Join-Path $root "v6\desktop-agent-rs"

Write-Host "Starting V6 Desktop Node" -ForegroundColor Green
Write-Host "Server=$ServerUrl Worker=$WorkerName Labels=$Labels" -ForegroundColor Cyan

while ($true) {
    try {
        if (Test-Path (Join-Path $agentPath "Cargo.toml")) {
            Push-Location $agentPath
            cargo run --release -- --server $ServerUrl --token $Token --name $WorkerName --labels $Labels --poll-sec $PollSec
            Pop-Location
        }
        else {
            throw "Rust desktop agent path not found"
        }
    }
    catch {
        Write-Host "Desktop node stopped: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "Restarting in 3 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        try { Pop-Location } catch {}
    }
}
