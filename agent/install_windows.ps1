param(
    [Parameter(Mandatory=$true)]
    [string]$ServerUrl,

    [string]$Token = "",
    [string]$AgentName = $env:COMPUTERNAME,
    [string]$Labels = ""
)

$ErrorActionPreference = "Stop"

$agentDir = "$env:ProgramData\bi-ide-agent"
New-Item -ItemType Directory -Path $agentDir -Force | Out-Null

$agentPy = Join-Path $agentDir "remote_worker_agent.py"
$runBat = Join-Path $agentDir "run_agent.bat"

$downloadUrl = "$($ServerUrl.TrimEnd('/'))/api/v1/orchestrator/download/agent.py"
Invoke-WebRequest -Uri $downloadUrl -OutFile $agentPy -UseBasicParsing

python -m pip install --upgrade requests | Out-Null

$tokenPart = ""
if ($Token -ne "") { $tokenPart = " --token `"$Token`"" }
$labelsPart = ""
if ($Labels -ne "") { $labelsPart = " --labels `"$Labels`"" }

$batContent = @"
@echo off
cd /d "$agentDir"
python "$agentPy" --server "$ServerUrl" --name "$AgentName"$labelsPart$tokenPart
"@
Set-Content -Path $runBat -Value $batContent -Encoding UTF8

$taskName = "BIIdeWorkerAgent"
$existing = schtasks /Query /TN $taskName 2>$null
if ($LASTEXITCODE -eq 0) {
    schtasks /Delete /TN $taskName /F | Out-Null
}

schtasks /Create /F /SC ONLOGON /RL HIGHEST /TN $taskName /TR "cmd /c \"$runBat\"" | Out-Null
schtasks /Run /TN $taskName | Out-Null

Write-Host "Agent installed and started. Task: $taskName"
