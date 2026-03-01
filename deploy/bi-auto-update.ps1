# BI-IDE v8 - Auto-Update Script (Windows)
# Runs every 2 minutes via Windows Scheduled Task
# Checks GitHub and auto-updates if new commits are found
#
# Install:   .\deploy\bi-auto-update.ps1 -Install
# Uninstall: .\deploy\bi-auto-update.ps1 -Uninstall
# Status:    .\deploy\bi-auto-update.ps1 -Status
# Dry Run:   .\deploy\bi-auto-update.ps1 -DryRun

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$DryRun,
    [switch]$Status,
    [string]$RepoDir = "C:\Users\BI\bi-ide-v8",
    [string]$Branch = "main",
    [string]$LogFile = "C:\Users\BI\bi-auto-update.log"
)

$ErrorActionPreference = "Stop"
$TaskName = "BI-IDE-AutoUpdate"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue
    Write-Host $line
}

function Rotate-Log {
    if (Test-Path $LogFile) {
        $size = (Get-Item $LogFile).Length
        if ($size -gt 5MB) {
            Move-Item $LogFile "$LogFile.old" -Force
            Write-Log "Log rotated"
        }
    }
}

# --- Install Scheduled Task ---
if ($Install) {
    Write-Host "=== Installing BI-IDE Auto-Update ==="

    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

    $scriptPath = $MyInvocation.MyCommand.Path
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`" -RepoDir `"$RepoDir`""

    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 2) -RepetitionDuration (New-TimeSpan -Days 9999)

    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable -MultipleInstances IgnoreNew

    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "BI-IDE Auto-Update: checks GitHub every 2 minutes" -RunLevel Highest

    Write-Host "[OK] Scheduled Task installed!"
    Write-Host "   Runs every 2 minutes"
    Write-Host "   Repo: $RepoDir"
    Write-Host "   Log:  $LogFile"
    exit 0
}

# --- Uninstall ---
if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "[OK] Scheduled Task removed"
    exit 0
}

# --- Status ---
if ($Status) {
    Write-Host "=== BI-IDE Auto-Update Status ==="
    Push-Location $RepoDir -ErrorAction SilentlyContinue
    if ($?) {
        $local = git rev-parse --short HEAD
        git fetch origin $Branch --quiet 2>$null
        $remote = git rev-parse --short "origin/$Branch" 2>$null
        Write-Host "Repo:   $RepoDir"
        Write-Host "Branch: $Branch"
        Write-Host "Local:  $local"
        Write-Host "Remote: $remote"

        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Host "Task:   $($task.State)"
        } else {
            Write-Host "Task:   NOT INSTALLED"
        }

        if (Test-Path $LogFile) {
            Write-Host "Last:   $(Get-Content $LogFile -Tail 1)"
        }
        Pop-Location
    } else {
        Write-Host "Repo not found at $RepoDir"
    }
    exit 0
}

# --- Dry Run ---
if ($DryRun) {
    Write-Host "=== Dry Run ==="
    Push-Location $RepoDir
    git fetch origin $Branch --quiet 2>$null
    $local = git rev-parse --short HEAD
    $remote = git rev-parse --short "origin/$Branch"
    if ($local -eq $remote) {
        Write-Host "[OK] Already up to date ($local)"
    } else {
        $behind = (git rev-list "HEAD..origin/$Branch" --count)
        Write-Host "[UPDATE] Available: $local -> $remote ($behind commits behind)"
        git log --oneline "HEAD..origin/$Branch"
    }
    Pop-Location
    exit 0
}

# === Main Update Logic ===
try {
    Rotate-Log

    # Check lock
    $lockFile = "$env:TEMP\bi-auto-update.lock"
    if (Test-Path $lockFile) {
        $lockAge = (Get-Date) - (Get-Item $lockFile).LastWriteTime
        if ($lockAge.TotalMinutes -lt 5) {
            exit 0
        }
        Remove-Item $lockFile -Force
    }
    New-Item $lockFile -ItemType File -Force | Out-Null

    # Check repo
    if (-not (Test-Path "$RepoDir\.git")) {
        Write-Log "Repo not found. Cloning..."
        git clone "https://github.com/hassanjava2/bi-ide-v8.git" $RepoDir
    }

    Push-Location $RepoDir

    # Fetch
    git fetch origin $Branch --quiet 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "[ERROR] git fetch failed"
        exit 1
    }

    # Compare
    $localSha = git rev-parse HEAD
    $remoteSha = git rev-parse "origin/$Branch"

    if ($localSha -eq $remoteSha) {
        Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
        Pop-Location
        exit 0
    }

    # Update!
    $behind = (git rev-list "HEAD..origin/$Branch" --count)
    Write-Log "[UPDATE] $behind commit(s) behind. Updating..."

    $rollbackSha = $localSha

    # Pull
    git pull origin $Branch --ff-only 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "[WARN] git pull failed, trying hard reset..."
        git reset --hard "origin/$Branch"
    }

    Write-Log "[OK] Code updated to $($remoteSha.Substring(0,8))"

    # Install Python deps
    if (Test-Path "$RepoDir\requirements.txt") {
        Write-Log "[DEPS] Installing dependencies..."
        pip install -r "$RepoDir\requirements.txt" --quiet 2>&1 | Select-Object -Last 2
    }

    # Restart NSSM services (if installed)
    foreach ($svc in @("bi-server", "bi-worker")) {
        $service = Get-Service -Name $svc -ErrorAction SilentlyContinue
        if ($service) {
            Write-Log "[RESTART] $svc..."
            Restart-Service -Name $svc -Force
            Start-Sleep -Seconds 3
            $service = Get-Service -Name $svc
            if ($service.Status -eq "Running") {
                Write-Log "[OK] $svc is running"
            } else {
                Write-Log "[ERROR] $svc failed! Rolling back..."
                git reset --hard $rollbackSha
                Restart-Service -Name $svc -Force -ErrorAction SilentlyContinue
            }
        }
    }

    $commitMsg = git log -1 --pretty="%s"
    Write-Log "[DONE] Update complete! $commitMsg"
    Pop-Location

}
catch {
    $errMsg = $_.ToString()
    Write-Log "[ERROR] $errMsg"
}
finally {
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}
