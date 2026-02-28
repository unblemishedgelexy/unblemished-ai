param(
  [string]$TaskName = "HumoniodServerAutopilot"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
  Write-Host ("Task not found: {0}" -f $TaskName) -ForegroundColor Yellow
  exit 0
}

try {
  Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}
catch {
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host ("Removed task: {0}" -f $TaskName) -ForegroundColor Green
