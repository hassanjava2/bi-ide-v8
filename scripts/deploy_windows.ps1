# ═══════════════════════════════════════════════════════════════════════════════
# BI-IDE v8 - سكربت النشر لنظام Windows (PowerShell)
# Windows Deployment Script with PowerShell
# ═══════════════════════════════════════════════════════════════════════════════

#Requires -Version 5.1

[CmdletBinding()]
param (
    [Parameter()]
    [ValidateSet('staging', 'production', 'local', 'all')]
    [string]$Environment = 'local',
    
    [Parameter()]
    [string]$Version = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    
    [Parameter()]
    [string]$Registry = 'ghcr.io',
    
    [Parameter()]
    [string]$ImageName = 'bi-ide',
    
    [Parameter()]
    [switch]$BuildOnly,
    
    [Parameter()]
    [switch]$PushOnly,
    
    [Parameter()]
    [switch]$DeployOnly,
    
    [Parameter()]
    [switch]$SkipTests,
    
    [Parameter()]
    [switch]$Rollback,
    
    [Parameter()]
    [switch]$HealthCheck,
    
    [Parameter()]
    [switch]$Help
)

# ═══════════════════════════════════════════════════════════════════════════════
# إعدادات الألوان
# ═══════════════════════════════════════════════════════════════════════════════
$Colors = @{
    Red = 'Red'
    Green = 'Green'
    Yellow = 'Yellow'
    Blue = 'Blue'
    Cyan = 'Cyan'
    White = 'White'
}

# ═══════════════════════════════════════════════════════════════════════════════
# الإعدادات العامة
# ═══════════════════════════════════════════════════════════════════════════════
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$BackupDir = Join-Path $ProjectRoot 'backups'
$LogDir = Join-Path $ProjectRoot 'logs'
$LogFile = Join-Path $LogDir "deploy_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# إعدادات البيئات
$Environments = @{
    Staging = @{
        Host = $env:STAGING_HOST
        ComposeFile = 'docker-compose.yml'
        Port = 8000
    }
    Production = @{
        Host = $env:PRODUCTION_HOST
        ComposeFile = 'docker-compose.prod.yml'
        Port = 8000
    }
    Local = @{
        Host = 'localhost'
        ComposeFile = 'docker-compose.yml'
        Port = 8000
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# دوال المساعدة
# ═══════════════════════════════════════════════════════════════════════════════

# طباعة رسالة ملونة
function Write-Log {
    param (
        [Parameter(Mandatory)]
        [ValidateSet('INFO', 'WARN', 'ERROR', 'DEBUG', 'SUCCESS')]
        [string]$Level,
        
        [Parameter(Mandatory)]
        [string]$Message,
        
        [switch]$NoTimestamp
    )
    
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $colorMap = @{
        'INFO' = 'Cyan'
        'WARN' = 'Yellow'
        'ERROR' = 'Red'
        'DEBUG' = 'Blue'
        'SUCCESS' = 'Green'
    }
    
    $color = $colorMap[$Level]
    $prefix = if ($NoTimestamp) { "[$Level]" } else { "[$Level] $timestamp -" }
    
    Write-Host $prefix -NoNewline -ForegroundColor $color
    Write-Host " $Message"
    
    # تسجيل في الملف
    "$prefix $Message" | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

# عرض شعار التطبيق
function Show-Banner {
    $banner = @"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗ ██╗     ██╗██████╗ ███████╗    ██╗██████╗ ███████╗                  ║
║   ██╔══██╗██║     ██║██╔══██╗██╔════╝    ██║██╔══██╗██╔════╝                  ║
║   ██████╔╝██║     ██║██║  ██║█████╗      ██║██║  ██║█████╗                    ║
║   ██╔══██╗██║     ██║██║  ██║██╔══╝      ██║██║  ██║██╔══╝                    ║
║   ██████╔╝███████╗██║██████╔╝███████╗    ██║██████╔╝███████╗                  ║
║   ╚═════╝ ╚══════╝╚═╝╚═════╝ ╚══════╝    ╚═╝╚═════╝ ╚══════╝                  ║
║                                                                               ║
║                      v8 - نظام النشر لـ Windows                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"@
    Write-Host $banner -ForegroundColor Cyan
}

# عرض المساعدة
function Show-Help {
    Show-Banner
    Write-Host @"
الاستخدام: .\deploy_windows.ps1 [خيارات]

الخيارات:
  -Environment    بيئة النشر (staging|production|local|all) [افتراضي: local]
  -Version        إصدار الصورة [افتراضي: timestamp]
  -Registry       سجل الحاويات [افتراضي: ghcr.io]
  -ImageName      اسم الصورة [افتراضي: bi-ide]
  -BuildOnly      بناء الصور فقط
  -PushOnly       دفع الصور فقط
  -DeployOnly     نشر فقط دون بناء
  -SkipTests      تخطي الاختبارات
  -Rollback       التراجع عن آخر نشر
  -HealthCheck    فحص صحة النظام فقط
  -Help           عرض هذه المساعدة

أمثلة:
  .\deploy_windows.ps1 -Environment local
  .\deploy_windows.ps1 -Environment staging -Version 1.2.3
  .\deploy_windows.ps1 -BuildOnly -Version 1.0.0
  .\deploy_windows.ps1 -HealthCheck
"@
}

# التحقق من المتطلبات
function Test-Prerequisites {
    Write-Log -Level 'INFO' -Message 'التحقق من المتطلبات...'
    
    $requiredCommands = @('docker', 'docker-compose')
    $missingCommands = @()
    
    foreach ($cmd in $requiredCommands) {
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
            $missingCommands += $cmd
        }
    }
    
    if ($missingCommands.Count -gt 0) {
        Write-Log -Level 'ERROR' -Message "الأوامر التالية غير موجودة: $($missingCommands -join ', ')"
        exit 1
    }
    
    # التحقق من تشغيل Docker
    try {
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Docker لا يعمل"
        }
    }
    catch {
        Write-Log -Level 'ERROR' -Message 'Docker لا يعمل. يرجى تشغيل Docker Desktop أولاً.'
        exit 1
    }
    
    # إنشاء المجلدات الضرورية
    @($BackupDir, $LogDir) | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ -Force | Out-Null
        }
    }
    
    Write-Log -Level 'SUCCESS' -Message '✓ جميع المتطلبات متوفرة'
}

# بناء صور Docker
function Build-Images {
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    Write-Log -Level 'INFO' -Message 'بدء بناء صور Docker...'
    Write-Log -Level 'INFO' -Message "الإصدار: $Version"
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    
    Push-Location $ProjectRoot
    
    try {
        # بناء صورة API
        Write-Log -Level 'INFO' -Message 'بناء صورة API...'
        docker build `
            --target runtime `
            -t "$Registry/$ImageName/api:$Version" `
            -t "$Registry/$ImageName/api:latest" `
            -f Dockerfile . 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            throw 'فشل بناء صورة API'
        }
        
        # بناء صورة Worker
        Write-Log -Level 'INFO' -Message 'بناء صورة Worker...'
        docker build `
            --target runtime `
            -t "$Registry/$ImageName/worker:$Version" `
            -t "$Registry/$ImageName/worker:latest" `
            -f Dockerfile . 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            throw 'فشل بناء صورة Worker'
        }
        
        Write-Log -Level 'SUCCESS' -Message '✓ تم بناء الصور بنجاح'
    }
    catch {
        Write-Log -Level 'ERROR' -Message $_.Exception.Message
        exit 1
    }
    finally {
        Pop-Location
    }
}

# دفع الصور إلى السجل
function Push-Images {
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    Write-Log -Level 'INFO' -Message "دفع الصور إلى السجل: $Registry"
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    
    # التحقق من تسجيل الدخول
    $dockerInfo = docker info 2>&1
    if ($dockerInfo -notmatch 'Username') {
        Write-Log -Level 'WARN' -Message 'غير مسجل الدخول إلى السجل. جاري تسجيل الدخول...'
        docker login $Registry
    }
    
    # دفع صور API
    Write-Log -Level 'INFO' -Message 'دفع صورة API...'
    docker push "$Registry/$ImageName/api:$Version" 2>&1 | Tee-Object -FilePath $LogFile -Append
    docker push "$Registry/$ImageName/api:latest" 2>&1 | Tee-Object -FilePath $LogFile -Append
    
    # دفع صور Worker
    Write-Log -Level 'INFO' -Message 'دفع صورة Worker...'
    docker push "$Registry/$ImageName/worker:$Version" 2>&1 | Tee-Object -FilePath $LogFile -Append
    docker push "$Registry/$ImageName/worker:latest" 2>&1 | Tee-Object -FilePath $LogFile -Append
    
    Write-Log -Level 'SUCCESS' -Message '✓ تم دفع الصور بنجاح'
}

# نشر محلي
function Deploy-Local {
    param ([string]$ComposeFile)
    
    Write-Log -Level 'INFO' -Message 'النشر على البيئة المحلية...'
    
    Push-Location $ProjectRoot
    
    try {
        # سحب أحدث الصور
        Write-Log -Level 'INFO' -Message 'سحب أحدث الصور...'
        docker-compose -f $ComposeFile pull 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        # إعادة تشغيل الخدمات
        Write-Log -Level 'INFO' -Message 'إعادة تشغيل الخدمات...'
        docker-compose -f $ComposeFile up -d --remove-orphans 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -ne 0) {
            throw 'فشل إعادة تشغيل الخدمات'
        }
        
        # تنظيف الصور القديمة
        Write-Log -Level 'INFO' -Message 'تنظيف الصور القديمة...'
        docker image prune -af --filter "until=168h" 2>&1 | Out-Null
        
        Write-Log -Level 'SUCCESS' -Message '✓ تم النشر المحلي بنجاح'
    }
    catch {
        Write-Log -Level 'ERROR' -Message $_.Exception.Message
        throw
    }
    finally {
        Pop-Location
    }
}

# فحص صحة النظام
function Test-Health {
    param (
        [string]$Host,
        [int]$Port = 8000
    )
    
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    Write-Log -Level 'INFO' -Message "فحص صحة النظام: $Host"
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    
    $maxRetries = 10
    $retryCount = 0
    $healthUrl = "http://$Host`:$Port/health"
    
    while ($retryCount -lt $maxRetries) {
        Write-Log -Level 'INFO' -Message "محاولة فحص الصحة رقم $($retryCount + 1)..."
        
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method GET -TimeoutSec 10 -ErrorAction Stop
            
            Write-Log -Level 'SUCCESS' -Message '✓ النظام يعمل بشكل صحيح!'
            Write-Log -Level 'INFO' -Message "استجابة API: $($response | ConvertTo-Json -Compress)"
            
            return $true
        }
        catch {
            $retryCount++
            if ($retryCount -lt $maxRetries) {
                Write-Log -Level 'WARN' -Message "فشل الاتصال، إعادة المحاولة خلال 10 ثوانٍ..."
                Start-Sleep -Seconds 10
            }
        }
    }
    
    Write-Log -Level 'ERROR' -Message "✗ فشل فحص الصحة بعد $maxRetries محاولات"
    return $false
}

# التراجع عن النشر
function Invoke-Rollback {
    param ([string]$ComposeFile)
    
    Write-Log -Level 'WARN' -Message '⚠️  جاري التراجع عن آخر نشر...'
    
    Push-Location $ProjectRoot
    
    try {
        docker-compose -f $ComposeFile down 2>&1 | Tee-Object -FilePath $LogFile -Append
        docker-compose -f $ComposeFile up -d 2>&1 | Tee-Object -FilePath $LogFile -Append
        
        Write-Log -Level 'SUCCESS' -Message '✓ تم التراجع بنجاح'
    }
    catch {
        Write-Log -Level 'ERROR' -Message "فشل التراجع: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }
}

# تشغيل الاختبارات
function Invoke-Tests {
    Write-Log -Level 'INFO' -Message 'تشغيل الاختبارات...'
    
    Push-Location $ProjectRoot
    
    try {
        # تشغيل pytest
        $testResult = & python -m pytest tests/ -v --tb=short 2>&1
        $testResult | Tee-Object -FilePath $LogFile -Append
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log -Level 'SUCCESS' -Message '✓ جميع الاختبارات ناجحة'
            return $true
        }
        else {
            Write-Log -Level 'ERROR' -Message '✗ فشلت بعض الاختبارات'
            return $false
        }
    }
    catch {
        Write-Log -Level 'ERROR' -Message "خطأ في تشغيل الاختبارات: $($_.Exception.Message)"
        return $false
    }
    finally {
        Pop-Location
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# الدالة الرئيسية
# ═══════════════════════════════════════════════════════════════════════════════
function Main {
    # عرض المساعدة
    if ($Help) {
        Show-Help
        return
    }
    
    # عرض الشعار
    Show-Banner
    
    # التحقق من المتطلبات
    Test-Prerequisites
    
    # فحص الصحة فقط
    if ($HealthCheck) {
        $envConfig = $Environments[$Environment]
        $result = Test-Health -Host $envConfig.Host -Port $envConfig.Port
        exit $result ? 0 : 1
    }
    
    # وضع التراجع
    if ($Rollback) {
        $envConfig = $Environments[$Environment]
        Invoke-Rollback -ComposeFile $envConfig.ComposeFile
        return
    }
    
    # البناء فقط
    if ($BuildOnly) {
        Build-Images
        return
    }
    
    # الدفع فقط
    if ($PushOnly) {
        Push-Images
        return
    }
    
    # النشر فقط
    if ($DeployOnly) {
        $envConfig = $Environments[$Environment]
        Deploy-Local -ComposeFile $envConfig.ComposeFile
        
        # فحص الصحة
        if (-not (Test-Health -Host $envConfig.Host -Port $envConfig.Port)) {
            Write-Log -Level 'ERROR' -Message 'فشل فحص الصحة!'
            exit 1
        }
        return
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # سير عمل النشر الكامل
    # ═══════════════════════════════════════════════════════════════════════════
    Write-Log -Level 'INFO' -Message 'بدء سير عمل النشر الكامل...'
    
    # تشغيل الاختبارات
    if (-not $SkipTests) {
        if (-not (Invoke-Tests)) {
            Write-Log -Level 'ERROR' -Message 'فشلت الاختبارات. إيقاف النشر.'
            exit 1
        }
    }
    
    # بناء الصور
    Build-Images
    
    # دفع الصور (للبيئات البعيدة فقط)
    if ($Environment -ne 'local') {
        Push-Images
    }
    
    # النشر
    $envConfig = $Environments[$Environment]
    if ($Environment -eq 'all') {
        # نشر على جميع البيئات
        foreach ($envName in @('Staging', 'Production')) {
            $config = $Environments[$envName]
            
            Write-Log -Level 'INFO' -Message "النشر على $envName..."
            Deploy-Local -ComposeFile $config.ComposeFile
            
            if (-not (Test-Health -Host $config.Host -Port $config.Port)) {
                Write-Log -Level 'ERROR' -Message "فشل النشر على $envName!"
                if ($envName -eq 'Production') {
                    Invoke-Rollback -ComposeFile $config.ComposeFile
                }
                exit 1
            }
            
            # انتظار قبل الإنتاج
            if ($envName -eq 'Staging') {
                Write-Log -Level 'WARN' -Message 'انتظر 30 ثانية قبل النشر في الإنتاج...'
                Start-Sleep -Seconds 30
            }
        }
    }
    else {
        # نشر على بيئة واحدة
        Deploy-Local -ComposeFile $envConfig.ComposeFile
        
        # فحص الصحة
        if (-not (Test-Health -Host $envConfig.Host -Port $envConfig.Port)) {
            Write-Log -Level 'ERROR' -Message 'فشل فحص الصحة! جاري التراجع...'
            Invoke-Rollback -ComposeFile $envConfig.ComposeFile
            exit 1
        }
    }
    
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    Write-Log -Level 'SUCCESS' -Message '✅ تم النشر بنجاح!'
    Write-Log -Level 'INFO' -Message '════════════════════════════════════════════════════════════'
    Write-Log -Level 'INFO' -Message "الإصدار: $Version"
    Write-Log -Level 'INFO' -Message "البيئة: $Environment"
    Write-Log -Level 'INFO' -Message "سجل النشر: $LogFile"
}

# ═══════════════════════════════════════════════════════════════════════════════
# تشغيل السكربت
# ═══════════════════════════════════════════════════════════════════════════════
try {
    Main
}
catch {
    Write-Log -Level 'ERROR' -Message "خطأ غير متوقع: $($_.Exception.Message)"
    exit 1
}
