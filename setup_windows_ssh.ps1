$ErrorActionPreference = 'Stop'

Write-Host '=== Setup OpenSSH Server on Windows ===' -ForegroundColor Cyan

# 1) Install OpenSSH Server capability if missing
$cap = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
if ($null -eq $cap) {
    throw 'OpenSSH.Server capability not found on this Windows build.'
}

if ($cap.State -ne 'Installed') {
    Write-Host 'Installing OpenSSH.Server...' -ForegroundColor Yellow
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
} else {
    Write-Host 'OpenSSH.Server already installed.' -ForegroundColor Green
}

# 2) Start and enable sshd service
Write-Host 'Enabling sshd service...' -ForegroundColor Yellow
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

# 3) Firewall rule for port 22
if (-not (Get-NetFirewallRule -Name sshd -ErrorAction SilentlyContinue)) {
    Write-Host 'Adding firewall rule for SSH (TCP 22)...' -ForegroundColor Yellow
    New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
} else {
    Write-Host 'Firewall rule sshd already exists.' -ForegroundColor Green
}

# 4) Print current status and IP addresses
Write-Host '\n=== SSH Service Status ===' -ForegroundColor Cyan
Get-Service sshd | Select-Object Name,Status,StartType | Format-Table -AutoSize

Write-Host '\n=== IPv4 Addresses ===' -ForegroundColor Cyan
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '169.254*' -and $_.IPAddress -ne '127.0.0.1' } |
    Select-Object InterfaceAlias,IPAddress |
    Format-Table -AutoSize

Write-Host '\nDone. From Mac connect using: ssh <WINDOWS_USER>@<IPv4>' -ForegroundColor Green
