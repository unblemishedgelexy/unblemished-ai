param(
  [string]$BaseUrl = "https://unblemished-ai.onrender.com",
  [string]$UserId = "team-train-user",
  [int]$DurationMinutes = 45,
  [int]$ChatDelaySeconds = 12,
  [int]$MonitorDelaySeconds = 20,
  [string]$AuthHeader = "",
  [string]$AuthValue = "",
  [string]$OutDir = "data/daily_updates",
  [switch]$NoNewWindows,
  [switch]$ForceStart
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

function Build-Headers {
  $headers = @{}
  if ($AuthHeader -and $AuthValue) {
    $headers[$AuthHeader] = $AuthValue
  }
  return $headers
}

function Invoke-ApiGetSafe {
  param(
    [string]$PathValue,
    [hashtable]$Headers
  )
  $uri = "{0}{1}" -f $BaseUrl.TrimEnd("/"), $PathValue
  try {
    $data = Invoke-RestMethod -Method GET -Uri $uri -Headers $Headers -TimeoutSec 30
    return @{ ok = $true; data = $data; error = "" }
  }
  catch {
    return @{ ok = $false; data = $null; error = $_.Exception.Message }
  }
}

function Get-Optional {
  param(
    [object]$ObjectValue,
    [string]$PropertyName,
    [object]$DefaultValue = $null
  )
  if ($null -eq $ObjectValue) { return $DefaultValue }
  if ($ObjectValue -is [System.Collections.IDictionary]) {
    if ($ObjectValue.Contains($PropertyName)) {
      $value = $ObjectValue[$PropertyName]
      if ($null -eq $value) { return $DefaultValue }
      return $value
    }
    return $DefaultValue
  }
  $prop = $ObjectValue.PSObject.Properties[$PropertyName]
  if ($null -eq $prop -or $null -eq $prop.Value) { return $DefaultValue }
  return $prop.Value
}

Ensure-Directory -PathValue $OutDir
$runId = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$runDir = Join-Path $OutDir ("team_train_{0}" -f $runId)
Ensure-Directory -PathValue $runDir

$headers = Build-Headers
$statusCheck = Invoke-ApiGetSafe -PathValue "/v1/system/status" -Headers $headers
$runtimeCheck = Invoke-ApiGetSafe -PathValue "/v1/system/runtime" -Headers $headers
if (-not $ForceStart.IsPresent) {
  if (-not $statusCheck.ok -or -not $runtimeCheck.ok) {
    Write-Host "Preflight failed: unable to reach status/runtime endpoints."
    Write-Host ("status_error={0}" -f $statusCheck.error)
    Write-Host ("runtime_error={0}" -f $runtimeCheck.error)
    Write-Host "Use -ForceStart only if you intentionally want to proceed."
    exit 1
  }
  $statusValue = [string](Get-Optional -ObjectValue $statusCheck.data -PropertyName "status" -DefaultValue "unknown")
  $memoryReady = (Get-Optional -ObjectValue $statusCheck.data -PropertyName "memory_ready" -DefaultValue $false) -eq $true
  $skillReady = (Get-Optional -ObjectValue $statusCheck.data -PropertyName "skill_engine_ready" -DefaultValue $false) -eq $true
  $effectiveBackend = [string](Get-Optional -ObjectValue $runtimeCheck.data -PropertyName "model_backend_effective" -DefaultValue "unknown")
  if ($statusValue -ne "ok" -or -not $memoryReady -or -not $skillReady) {
    Write-Host "Preflight blocked: server is not fully ready for team training."
    Write-Host ("status={0}" -f $statusValue)
    Write-Host ("memory_ready={0}" -f $memoryReady)
    Write-Host ("skill_engine_ready={0}" -f $skillReady)
    Write-Host ("backend_effective={0}" -f $effectiveBackend)
    Write-Host "Fix server health first, or pass -ForceStart to bypass."
    exit 1
  }
}

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
