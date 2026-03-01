param(
  [string]$BaseUrl = "https://unblemished-ai.onrender.com",
  [string]$UserId = "team-train-user",
  [int]$DurationMinutes = 45,
  [int]$ChatDelaySeconds = 12,
  [int]$MonitorDelaySeconds = 20,
  [string]$AuthHeader = "",
  [string]$AuthValue = "",
  [string]$OutDir = "data/daily_updates",
  [switch]$NoNewWindows
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($DurationMinutes -lt 1) { throw "DurationMinutes must be >= 1" }
if ($ChatDelaySeconds -lt 0) { throw "ChatDelaySeconds must be >= 0" }
if ($MonitorDelaySeconds -lt 0) { throw "MonitorDelaySeconds must be >= 0" }

function Ensure-Directory {
  param([string]$PathValue)
  if (-not (Test-Path $PathValue)) {
    New-Item -Path $PathValue -ItemType Directory | Out-Null
  }
}

Ensure-Directory -PathValue $OutDir
$runId = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$runDir = Join-Path $OutDir ("team_train_{0}" -f $runId)
Ensure-Directory -PathValue $runDir

$stopFlagPath = Join-Path $runDir "STOP.flag"
$workerScriptPath = Join-Path $PSScriptRoot "team_train_worker.ps1"
if (-not (Test-Path $workerScriptPath)) {
  throw "Worker script not found: $workerScriptPath"
}

$roles = @(
  "chat_general",
  "chat_technical",
  "chat_companion",
  "status_runtime",
  "memory_skills"
)

Write-Host ""
Write-Host ("Team Train Run ID: {0}" -f $runId)
Write-Host ("BaseUrl: {0}" -f $BaseUrl)
Write-Host ("RunDir: {0}" -f $runDir)
Write-Host ("StopFlag: {0}" -f $stopFlagPath)
Write-Host ""

foreach ($role in $roles) {
  $roleDelay = $ChatDelaySeconds
  if ($role -in @("status_runtime", "memory_skills")) {
    $roleDelay = $MonitorDelaySeconds
  }
  $logPath = Join-Path $runDir ("{0}.jsonl" -f $role)
  $argList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $workerScriptPath,
    "-BaseUrl", $BaseUrl,
    "-Role", $role,
    "-UserId", $UserId,
    "-DurationMinutes", "$DurationMinutes",
    "-DelaySeconds", "$roleDelay",
    "-LogPath", $logPath,
    "-StopFlagPath", $stopFlagPath
  )
  if ($AuthHeader -and $AuthValue) {
    $argList += @("-AuthHeader", $AuthHeader, "-AuthValue", $AuthValue)
  }
  if ($NoNewWindows.IsPresent) {
    Start-Job -Name ("team-{0}" -f $role) -ScriptBlock {
      param([string[]]$Arguments)
      & powershell @Arguments
    } -ArgumentList (, $argList) | Out-Null
    Write-Host ("Started background job for role={0}" -f $role)
  }
  else {
    Start-Process -FilePath "powershell" -ArgumentList $argList -WindowStyle Normal | Out-Null
    Write-Host ("Started new terminal for role={0}" -f $role)
  }
}

Write-Host ""
Write-Host "5-route team training launched."
Write-Host ("To stop all workers, create stop flag:")
Write-Host ("New-Item -Path `"{0}`" -ItemType File -Force" -f $stopFlagPath)
Write-Host ("Logs: {0}" -f $runDir)
