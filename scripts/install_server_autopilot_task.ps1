param(
  [string]$TaskName = "HumoniodServerAutopilot",
  [string]$OnlineFrom = "09:00",
  [string]$OnlineTo = "23:30",
  [ValidateSet("heuristic", "local_light")] [string]$OfflineMode = "heuristic"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$autopilotScript = Join-Path $repoRoot "scripts\server_autopilot.ps1"

if (-not (Test-Path $autopilotScript)) {
  throw "Autopilot script not found: $autopilotScript"
}

$arguments = @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", ('"{0}"' -f $autopilotScript),
  "-OnlineFrom", $OnlineFrom,
  "-OnlineTo", $OnlineTo,
  "-OfflineMode", $OfflineMode
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments -WorkingDirectory $repoRoot
$triggerStartup = New-ScheduledTaskTrigger -AtStartup
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 3650)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger @($triggerStartup, $triggerLogon) `
  -Settings $settings `
  -Description "Humoniod AI server autopilot (self-heal + online-hours profile switching)"

Start-ScheduledTask -TaskName $TaskName

Write-Host ("Installed and started task: {0}" -f $TaskName) -ForegroundColor Green
Write-Host ("Schedule online window: {0} - {1}" -f $OnlineFrom, $OnlineTo)
Write-Host ("Offline mode: {0}" -f $OfflineMode)
