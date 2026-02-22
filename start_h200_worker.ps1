param(
    [string]$ApiUrl = $env:BI_IDE_API_URL,
    [string]$WorkerId = $env:WORKER_ID,
    [int]$PollSec = 2,
    [int]$TrainSec = 30
)

if (-not $ApiUrl -or $ApiUrl.Trim() -eq "") {
    $ApiUrl = "http://localhost:8000"
}

if (-not $WorkerId -or $WorkerId.Trim() -eq "") {
    $WorkerId = "h200-$env:COMPUTERNAME"
}

$env:BI_IDE_API_URL = $ApiUrl
$env:WORKER_ID = $WorkerId
$env:WORKER_POLL_SEC = "$PollSec"
$env:WORKER_TRAIN_SEC = "$TrainSec"

Write-Host "Starting resilient worker" -ForegroundColor Green
Write-Host "API=$ApiUrl WORKER=$WorkerId POLL=$PollSec TRAIN=$TrainSec" -ForegroundColor Cyan

while ($true) {
    try {
        D:/bi-ide-v8/.venv/Scripts/python.exe distributed_worker_agent.py --api $ApiUrl --worker-id $WorkerId --poll-sec $PollSec --train-sec $TrainSec
        Write-Host "Worker exited; restarting in 3 seconds..." -ForegroundColor Yellow
    }
    catch {
        Write-Host "Worker crashed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Restarting in 3 seconds..." -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 3
}
