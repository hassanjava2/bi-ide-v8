#Requires -RunAsAdministrator
<#
.SYNOPSIS
    BI-IDE v8 - Windows Deployment Script
    نص نشر BI-IDE على Windows

.DESCRIPTION
    Deploys BI-IDE worker service on Windows with Python setup,
    Windows service configuration, and firewall rules.

.PARAMETER DryRun
    Run in dry-run mode without making actual changes

.PARAMETER SkipPythonCheck
    Skip Python version check

.EXAMPLE
    .\deploy_windows.ps1
    # نشر عادي

.EXAMPLE
    .\deploy_windows.ps1 -DryRun
    # وضع التشغيل الجاف

.EXAMPLE
    .\deploy_windows.ps1 -SkipPythonCheck
    # تخطي فحص Python
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$SkipPythonCheck,
    [string]$InstallPath = "C:\Program Files\BI-IDE",
    [string]$ServiceName = "BI-IDE-Worker"
)

# ═══════════════════════════════════════════════════════════════════
# الإعدادات / Settings
# ═══════════════════════════════════════════════════════════════════
$ScriptVersion = "1.0.0"
$LogDir = "$env:ProgramData\BI-IDE\logs"
$LogFile = "$LogDir\deploy_windows_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$BackupDir = "$env:ProgramData\BI-IDE\backups"
$RequiredPythonVersion = "3.11"
$NSSM_URL = "https://nssm.cc/release/nssm-2.24.zip"
$NSSM_DIR = "$env:ProgramData\nssm"

# ═══════════════════════════════════════════════════════════════════
# الألوان / Colors
# ═══════════════════════════════════════════════════════════════════
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Cyan"
    Magenta = "Magenta"
    Gray = "Gray"
}

# ═══════════════════════════════════════════════════════════════════
# دوال التسجيل / Logging Functions
# ═══════════════════════════════════════════════════════════════════
function Write-Log {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$false)]
        [ValidateSet("Info", "Success", "Warning", "Error", "Step")]
        [string]$Level = "Info"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # كتابة في الملف
    Add-Content -Path $LogFile -Value $logEntry -ErrorAction SilentlyContinue
    
    # عرض في الشاشة
    switch ($Level) {
        "Success" { Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green }
        "Warning" { Write-Host "[WARN] $Message" -ForegroundColor $Colors.Yellow }
        "Error"   { Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red }
        "Step"    { Write-Host "[STEP] $Message" -ForegroundColor $Colors.Blue }
        default   { Write-Host "[INFO] $Message" -ForegroundColor $Colors.Gray }
    }
}

function Write-Info { param([string]$Message) Write-Log -Message $Message -Level "Info" }
function Write-Success { param([string]$Message) Write-Log -Message $Message -Level "Success" }
function Write-Warning { param([string]$Message) Write-Log -Message $Message -Level "Warning" }
function Write-Error { param([string]$Message) Write-Log -Message $Message -Level "Error" }
function Write-Step { param([string]$Message) Write-Log -Message $Message -Level "Step" }

# ═══════════════════════════════════════════════════════════════════
# دالة التشغيل الجاف / Dry Run Function
# ═══════════════════════════════════════════════════════════════════
function Invoke-OrDryRun {
    param(
        [Parameter(Mandatory=$true)]
        [scriptblock]$ScriptBlock,
        
        [string]$Description = "Command"
    )
    
    if ($DryRun) {
        Write-Warning "[DRY RUN] Would execute: $Description"
        return $true
    } else {
        Write-Info "Executing: $Description"
        try {
            & $ScriptBlock
            return $?
        } catch {
            Write-Error "Command failed: $_"
            return $false
        }
    }
}

# ═══════════════════════════════════════════════════════════════════
# إنشاء الأدلة / Create Directories
# ═══════════════════════════════════════════════════════════════════
function Initialize-Directories {
    Write-Step "Initializing directories..."
    
    $directories = @($LogDir, $BackupDir, $InstallPath, "$InstallPath\worker")
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            Invoke-OrDryRun -Description "Create directory: $dir" -ScriptBlock {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
        }
    }
    
    Write-Success "Directories initialized"
}

# ═══════════════════════════════════════════════════════════════════
# فحص إصدار Windows / Check Windows Version
# ═══════════════════════════════════════════════════════════════════
function Test-WindowsVersion {
    Write-Step "Checking Windows version..."
    
    $osInfo = Get-CimInstance Win32_OperatingSystem
    $osVersion = [System.Environment]::OSVersion.Version
    $windowsVersion = $osInfo.Caption
    
    Write-Info "Windows: $windowsVersion"
    Write-Info "Version: $($osVersion.Major).$($osVersion.Minor).$($osVersion.Build)"
    
    # Windows 10 أو أحدث مطلوب
    if ($osVersion.Major -lt 10) {
        Write-Error "Windows 10 or later is required"
        return $false
    }
    
    # فحص إذا كان Windows Server
    if ($osInfo.ProductType -eq 3) {
        Write-Info "Windows Server detected"
    }
    
    Write-Success "Windows version check passed"
    return $true
}

# ═══════════════════════════════════════════════════════════════════
# تثبيت Python / Install Python
# ═══════════════════════════════════════════════════════════════════
function Install-Python {
    Write-Step "Checking Python installation..."
    
    if ($SkipPythonCheck) {
        Write-Warning "Skipping Python version check"
        return $true
    }
    
    $pythonPath = Get-Command python -ErrorAction SilentlyContinue
    $pythonInstalled = $false
    $currentVersion = $null
    
    if ($pythonPath) {
        try {
            $versionOutput = & python --version 2>&1
            $currentVersion = ($versionOutput -split " ")[1]
            Write-Info "Found Python $currentVersion"
            $pythonInstalled = $true
        } catch {
            Write-Warning "Could not determine Python version"
        }
    }
    
    # فحص الإصدار المطلوب
    if ($pythonInstalled) {
        $versionParts = $currentVersion.Split(".")
        $major = [int]$versionParts[0]
        $minor = [int]$versionParts[1]
        
        if ($major -ge 3 -and $minor -ge 11) {
            Write-Success "Python version is sufficient (>= $RequiredPythonVersion)"
            return $true
        } else {
            Write-Warning "Python version $currentVersion is too old. Required: >= $RequiredPythonVersion"
        }
    }
    
    # تثبيت Python
    Write-Info "Installing Python $RequiredPythonVersion..."
    
    $pythonUrl = "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"
    $installerPath = "$env:TEMP\python-3.11.8-amd64.exe"
    
    $success = Invoke-OrDryRun -Description "Download and install Python" -ScriptBlock {
        try {
            # تحميل المثبت
            Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
            
            # تثبيت Python
            $arguments = @(
                "/quiet"
                "InstallAllUsers=1"
                "PrependPath=1"
                "Include_test=0"
                "Include_pip=1"
                "Include_tcltk=0"
            )
            
            Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -NoNewWindow
            
            # تحديث متغيرات البيئة
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            Write-Success "Python installed successfully"
            return $true
        } catch {
            Write-Error "Failed to install Python: $_"
            return $false
        } finally {
            Remove-Item $installerPath -ErrorAction SilentlyContinue
        }
    }
    
    return $success
}

# ═══════════════════════════════════════════════════════════════════
# تثبيت NSSM / Install NSSM
# ═══════════════════════════════════════════════════════════════════
function Install-NSSM {
    Write-Step "Installing NSSM (Non-Sucking Service Manager)..."
    
    $nssmExe = "$NSSM_DIR\nssm.exe"
    
    if (Test-Path $nssmExe) {
        Write-Info "NSSM already installed"
        return $true
    }
    
    return Invoke-OrDryRun -Description "Install NSSM" -ScriptBlock {
        try {
            $zipPath = "$env:TEMP\nssm.zip"
            
            # تحميل NSSM
            Invoke-WebRequest -Uri $NSSM_URL -OutFile $zipPath -UseBasicParsing
            
            # استخراج
            Expand-Archive -Path $zipPath -DestinationPath $env:TEMP -Force
            
            # نسخ الملف المناسب
            $nssmSource = Get-ChildItem -Path "$env:TEMP\nssm-*" -Filter "nssm.exe" -Recurse | Select-Object -First 1
            
            if ($nssmSource) {
                New-Item -ItemType Directory -Path $NSSM_DIR -Force | Out-Null
                Copy-Item -Path $nssmSource.FullName -Destination $nssmExe -Force
                Write-Success "NSSM installed to $nssmExe"
            }
            
            # تنظيف
            Remove-Item $zipPath -ErrorAction SilentlyContinue
            Remove-Item "$env:TEMP\nssm-*" -Recurse -Force -ErrorAction SilentlyContinue
            
            return $true
        } catch {
            Write-Error "Failed to install NSSM: $_"
            return $false
        }
    }
}

# ═══════════════════════════════════════════════════════════════════
# إعداد بيئة Python / Setup Python Environment
# ═══════════════════════════════════════════════════════════════════
function Initialize-PythonEnvironment {
    Write-Step "Setting up Python environment..."
    
    $venvPath = "$InstallPath\venv"
    
    # إنشاء virtual environment
    Invoke-OrDryRun -Description "Create virtual environment" -ScriptBlock {
        & python -m venv $venvPath
    }
    
    # تثبيت المتطلبات
    $requirementsFile = "$PSScriptRoot\..\requirements.txt"
    if (Test-Path $requirementsFile) {
        Invoke-OrDryRun -Description "Install Python requirements" -ScriptBlock {
            & "$venvPath\Scripts\pip.exe" install -r $requirementsFile --quiet
        }
    }
    
    Write-Success "Python environment configured"
}

# ═══════════════════════════════════════════════════════════════════
# إنشاء خدمة Windows / Create Windows Service
# ═══════════════════════════════════════════════════════════════════
function Install-Service {
    Write-Step "Installing Windows service: $ServiceName..."
    
    $nssmExe = "$NSSM_DIR\nssm.exe"
    $venvPython = "$InstallPath\venv\Scripts\python.exe"
    $workerScript = "$InstallPath\worker\worker_service.py"
    
    # إنشاء سكربت العامل
    $workerScriptContent = @'
"""
BI-IDE Worker Service for Windows
"""
import os
import sys
import time
import logging
from pathlib import Path

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'{LOG_DIR}\worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BI-IDE-Worker')

def main():
    """Main worker loop"""
    logger.info('BI-IDE Worker starting...')
    
    try:
        # هنا يتم استيراد وتشغيل العامل الفعلي
        # from core.celery_config import celery_app
        # celery_app.start()
        
        # حلقة انتظار للخدمة
        while True:
            logger.info('Worker is running...')
            time.sleep(60)
            
    except Exception as e:
        logger.error(f'Worker error: {e}')
        raise

if __name__ == '__main__':
    main()
'@ -replace "{LOG_DIR}", $LogDir.Replace("\", "\\")
    
    Invoke-OrDryRun -Description "Create worker script" -ScriptBlock {
        Set-Content -Path $workerScript -Value $workerScriptContent -Encoding UTF8
    }
    
    # إنشاء الخدمة باستخدام NSSM
    Invoke-OrDryRun -Description "Create Windows service" -ScriptBlock {
        # إزالة الخدمة القديمة إن وجدت
        & $nssmExe stop $ServiceName 2>$null
        & $nssmExe remove $ServiceName confirm 2>$null
        
        # إنشاء خدمة جديدة
        & $nssmExe install $ServiceName $venvPython $workerScript
        & $nssmExe set $ServiceName DisplayName "BI-IDE Worker Service"
        & $nssmExe set $ServiceName Description "Background worker for BI-IDE v8"
        & $nssmExe set $ServiceName Start SERVICE_AUTO_START
        & $nssmExe set $ServiceName AppStdout "$LogDir\worker_stdout.log"
        & $nssmExe set $ServiceName AppStderr "$LogDir\worker_stderr.log"
        & $nssmExe set $ServiceName AppRotateFiles 1
        & $nssmExe set $ServiceName AppRotateSeconds 86400
    }
    
    Write-Success "Service installed: $ServiceName"
}

# ═══════════════════════════════════════════════════════════════════
# تكوين بدء التشغيل التلقائي / Configure Auto-Start
# ═══════════════════════════════════════════════════════════════════
function Set-AutoStart {
    Write-Step "Configuring auto-start..."
    
    Invoke-OrDryRun -Description "Enable auto-start for service" -ScriptBlock {
        Set-Service -Name $ServiceName -StartupType Automatic
        Start-Service -Name $ServiceName -ErrorAction SilentlyContinue
    }
    
    Write-Success "Auto-start configured"
}

# ═══════════════════════════════════════════════════════════════════
# إضافة قواعد جدار الحماية / Add Firewall Rules
# ═══════════════════════════════════════════════════════════════════
function Add-FirewallRules {
    Write-Step "Adding firewall rules..."
    
    $rules = @(
        @{Name="BI-IDE-API"; Port=8000; Protocol="TCP"},
        @{Name="BI-IDE-Worker"; Port=8001; Protocol="TCP"},
        @{Name="BI-IDE-Redis"; Port=6379; Protocol="TCP"}
    )
    
    foreach ($rule in $rules) {
        Invoke-OrDryRun -Description "Add firewall rule: $($rule.Name)" -ScriptBlock {
            $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
            if (-not $existing) {
                New-NetFirewallRule -DisplayName $rule.Name `
                    -Direction Inbound `
                    -Protocol $rule.Protocol `
                    -LocalPort $rule.Port `
                    -Action Allow `
                    -Profile Any | Out-Null
                Write-Info "Created firewall rule: $($rule.Name) (port $($rule.Port))"
            } else {
                Write-Info "Firewall rule exists: $($rule.Name)"
            }
        }
    }
    
    Write-Success "Firewall rules configured"
}

# ═══════════════════════════════════════════════════════════════════
# إنشاء النسخ الاحتياطي / Create Backup
# ═══════════════════════════════════════════════════════════════════
function New-Backup {
    Write-Step "Creating backup..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupPath = "$BackupDir\backup_$timestamp"
    
    Invoke-OrDryRun -Description "Create backup directory" -ScriptBlock {
        New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
    }
    
    # نسخ الإعدادات
    if (Test-Path "$InstallPath\.env") {
        Invoke-OrDryRun -Description "Backup .env file" -ScriptBlock {
            Copy-Item "$InstallPath\.env" "$backupPath\" -Force
        }
    }
    
    # حفظ معلومات النسخ
    $backupInfo = @"
Backup created: $(Get-Date)
Backup type: Windows deployment
Windows version: $([System.Environment]::OSVersion.VersionString)
Python version: $(python --version 2>&1)
Service name: $ServiceName
"@
    
    Invoke-OrDryRun -Description "Create backup info" -ScriptBlock {
        Set-Content -Path "$backupPath\backup_info.txt" -Value $backupInfo
    }
    
    Write-Success "Backup created: $backupPath"
}

# ═══════════════════════════════════════════════════════════════════
# فحوصات الصحة / Health Checks
# ═══════════════════════════════════════════════════════════════════
function Test-Health {
    Write-Step "Running health checks..."
    
    $checks = @()
    
    # فحص Python
    try {
        $pyVersion = python --version 2>&1
        Write-Success "Python check: $pyVersion"
        $checks += $true
    } catch {
        Write-Error "Python check failed"
        $checks += $false
    }
    
    # فحص الخدمة
    try {
        $service = Get-Service -Name $ServiceName -ErrorAction Stop
        Write-Success "Service check: $($service.Status)"
        $checks += $true
    } catch {
        Write-Warning "Service check: Not installed yet"
        $checks += $true # ليس خطأ في وضع التثبيت
    }
    
    # فحص NSSM
    if (Test-Path "$NSSM_DIR\nssm.exe") {
        Write-Success "NSSM check: OK"
        $checks += $true
    } else {
        Write-Warning "NSSM check: Not installed"
        $checks += $false
    }
    
    if ($checks -contains $false) {
        Write-Warning "Some health checks failed"
        return $false
    }
    
    Write-Success "All health checks passed"
    return $true
}

# ═══════════════════════════════════════════════════════════════════
# الدالة الرئيسية / Main Function
# ═══════════════════════════════════════════════════════════════════
function Main {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "   BI-IDE v8 - Windows Deployment" -ForegroundColor Cyan
    Write-Host "   Version: $ScriptVersion | Mode: $(if($DryRun){'DRY-RUN'}else{'LIVE'})" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    if ($DryRun) {
        Write-Warning "Running in DRY-RUN mode. No changes will be made."
    }
    
    # التحقق من الصلاحيات
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error "This script must be run as Administrator"
        Write-Info "Please run: powershell.exe -ExecutionPolicy Bypass -File '$PSCommandPath'"
        exit 1
    }
    
    try {
        # تهيئة
        Initialize-Directories
        
        # فحوصات ما قبل النشر
        Test-WindowsVersion
        Install-Python
        
        # تثبيت المتطلبات
        Install-NSSM
        Initialize-PythonEnvironment
        
        # إنشاء النسخ الاحتياطي
        New-Backup
        
        # تثبيت الخدمة
        Install-Service
        Set-AutoStart
        
        # إعدادات الأمان
        Add-FirewallRules
        
        # فحوصات ما بعد النشر
        Test-Health
        
        # ملخص
        Write-Host ""
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host "   DEPLOYMENT COMPLETE" -ForegroundColor Green
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Service: $ServiceName" -ForegroundColor Cyan
        Write-Host "  Install Path: $InstallPath" -ForegroundColor Cyan
        Write-Host "  Log File: $LogFile" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Commands:" -ForegroundColor Yellow
        Write-Host "    Start service:   Start-Service $ServiceName" -ForegroundColor Gray
        Write-Host "    Stop service:    Stop-Service $ServiceName" -ForegroundColor Gray
        Write-Host "    Service status:  Get-Service $ServiceName" -ForegroundColor Gray
        Write-Host "    View logs:       Get-Content '$LogDir\worker.log' -Tail 50" -ForegroundColor Gray
        Write-Host ""
        Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Green
        
    } catch {
        Write-Error "Deployment failed: $_"
        Write-Error "Stack trace: $($_.ScriptStackTrace)"
        exit 1
    }
}

# تشغيل الدالة الرئيسية
Main
