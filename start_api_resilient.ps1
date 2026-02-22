param(
    [int]$PreferredPort = 8000,
    [switch]$KillIfBusy
)

function Test-PortBusy([int]$Port) {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    return $conn
}

$selectedPort = $PreferredPort
$busy = Test-PortBusy -Port $selectedPort

if ($busy -and $KillIfBusy.IsPresent) {
    try {
        Stop-Process -Id $busy.OwningProcess -Force -ErrorAction Stop
        Start-Sleep -Milliseconds 500
        $busy = Test-PortBusy -Port $selectedPort
    }
    catch {
        Write-Host "Failed to kill process on port $selectedPort: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

if ($busy) {
    for ($p = $PreferredPort + 1; $p -le $PreferredPort + 20; $p++) {
        if (-not (Test-PortBusy -Port $p)) {
            $selectedPort = $p
            break
        }
    }
}

$env:PORT = "$selectedPort"
Write-Host "Starting API on port $selectedPort" -ForegroundColor Green

D:/bi-ide-v8/.venv/Scripts/python.exe -m api.app
