param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$UserId = "dashboard-user",
  [int]$DurationHours = 12,
  [int]$PollMinutes = 15,
  [string]$AuthHeader = "",
  [string]$AuthValue = "",
  [string]$OutDir = "data/daily_updates",
  [switch]$RecordEveryPoll,
  [switch]$RunOnce
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($DurationHours -lt 1) {
  throw "DurationHours must be >= 1"
}
if ($PollMinutes -lt 1) {
  throw "PollMinutes must be >= 1"
}

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

function Invoke-SystemApi {
  param(
    [string]$PathValue,
    [hashtable]$Headers
  )
  $uri = "{0}{1}" -f $BaseUrl.TrimEnd("/"), $PathValue
  return Invoke-RestMethod -Method GET -Uri $uri -Headers $Headers -TimeoutSec 20
}

function Get-ValueOrDefault {
  param(
    [object]$ObjectValue,
    [string]$PropertyName,
    [object]$DefaultValue = $null
  )
  if ($null -eq $ObjectValue) {
    return $DefaultValue
  }
  $prop = $ObjectValue.PSObject.Properties[$PropertyName]
  if ($null -eq $prop -or $null -eq $prop.Value) {
    return $DefaultValue
  }
  return $prop.Value
}

function Compute-Health {
  param([object]$Status)
  $issues = @()
  if ((Get-ValueOrDefault -ObjectValue $Status -PropertyName "model_ready" -DefaultValue $false) -ne $true) {
    $issues += "model_not_ready"
  }
  if ((Get-ValueOrDefault -ObjectValue $Status -PropertyName "memory_ready" -DefaultValue $false) -ne $true) {
    $issues += "memory_not_ready"
  }
  if ((Get-ValueOrDefault -ObjectValue $Status -PropertyName "skill_engine_ready" -DefaultValue $false) -ne $true) {
    $issues += "skill_engine_not_ready"
  }
  if ((Get-ValueOrDefault -ObjectValue $Status -PropertyName "tool_engine_ready" -DefaultValue $false) -ne $true) {
    $issues += "tool_engine_not_ready"
  }
  if ($issues.Count -eq 0) {
    return @{ health = "OK"; issues = @() }
  }
  return @{ health = "DEGRADED"; issues = $issues }
}

function Build-Snapshot {
  param(
    [object]$Runtime,
    [object]$Status,
    [object]$Memory,
    [object]$Skills,
    [object]$Analyzer
  )

  $healthObj = Compute-Health -Status $Status
  $telemetry = Get-ValueOrDefault -ObjectValue $Status -PropertyName "telemetry" -DefaultValue $null

  return [ordered]@{
    at_utc = (Get-Date).ToUniversalTime().ToString("o")
    health = $healthObj.health
    issues = $healthObj.issues
    backend_configured = (Get-ValueOrDefault -ObjectValue $Runtime -PropertyName "model_backend_configured" -DefaultValue "unknown")
    backend_effective = (Get-ValueOrDefault -ObjectValue $Runtime -PropertyName "model_backend_effective" -DefaultValue "unknown")
    model_ready = (Get-ValueOrDefault -ObjectValue $Status -PropertyName "model_ready" -DefaultValue $false)
    memory_ready = (Get-ValueOrDefault -ObjectValue $Status -PropertyName "memory_ready" -DefaultValue $false)
    skill_engine_ready = (Get-ValueOrDefault -ObjectValue $Status -PropertyName "skill_engine_ready" -DefaultValue $false)
    tool_engine_ready = (Get-ValueOrDefault -ObjectValue $Status -PropertyName "tool_engine_ready" -DefaultValue $false)
    request_count = (Get-ValueOrDefault -ObjectValue $telemetry -PropertyName "request_count" -DefaultValue 0)
    fallback_rate = (Get-ValueOrDefault -ObjectValue $telemetry -PropertyName "fallback_rate" -DefaultValue 0)
    latency_avg_ms = (Get-ValueOrDefault -ObjectValue $telemetry -PropertyName "latency_avg_ms" -DefaultValue 0)
    memory_entries = (Get-ValueOrDefault -ObjectValue $Memory -PropertyName "memory_entries_count" -DefaultValue 0)
    embedding_provider = (Get-ValueOrDefault -ObjectValue $Memory -PropertyName "embedding_provider" -DefaultValue "unknown")
    skills_total = (Get-ValueOrDefault -ObjectValue $Skills -PropertyName "total_count" -DefaultValue 0)
    skills_active = (Get-ValueOrDefault -ObjectValue $Skills -PropertyName "active_count" -DefaultValue 0)
    analyzer_enabled = (Get-ValueOrDefault -ObjectValue $Analyzer -PropertyName "analyzer_enabled" -DefaultValue $false)
    analyzer_running = (Get-ValueOrDefault -ObjectValue $Analyzer -PropertyName "analyzer_running" -DefaultValue $false)
    analyzer_entry_count = (Get-ValueOrDefault -ObjectValue $Analyzer -PropertyName "entry_count" -DefaultValue 0)
  }
}

function Build-Signature {
  param([hashtable]$Snapshot)
  $fields = @(
    $Snapshot.health,
    ($Snapshot.issues -join ","),
    $Snapshot.backend_configured,
    $Snapshot.backend_effective,
    $Snapshot.model_ready,
    $Snapshot.memory_ready,
    $Snapshot.skill_engine_ready,
    $Snapshot.tool_engine_ready,
    $Snapshot.fallback_rate,
    $Snapshot.memory_entries,
    $Snapshot.skills_total,
    $Snapshot.skills_active
  )
  return ($fields -join "|")
}

function Write-Line {
  param([string]$Message)
  Write-Host ("[{0}] {1}" -f (Get-Date).ToString("HH:mm:ss"), $Message)
}

Ensure-Directory -PathValue $OutDir
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$jsonlPath = Join-Path $OutDir ("full_day_update_{0}.jsonl" -f $stamp)
$summaryPath = Join-Path $OutDir ("full_day_summary_{0}.txt" -f $stamp)

$headers = Build-Headers
$startUtc = (Get-Date).ToUniversalTime()
$deadline = $startUtc.AddHours($DurationHours)
$iteration = 0
$recorded = 0
$lastSignature = ""
$healthDegradedCount = 0
$errorCount = 0

Write-Line ("Full-day monitor started. BaseUrl={0}, UserId={1}, DurationHours={2}, PollMinutes={3}" -f $BaseUrl, $UserId, $DurationHours, $PollMinutes)
Write-Line ("Output JSONL: {0}" -f $jsonlPath)

while ((Get-Date).ToUniversalTime() -lt $deadline) {
  $iteration += 1
  try {
    $runtime = Invoke-SystemApi -PathValue "/v1/system/runtime" -Headers $headers
    $status = Invoke-SystemApi -PathValue "/v1/system/status" -Headers $headers
    $memory = Invoke-SystemApi -PathValue "/v1/system/memory" -Headers $headers
    $skills = Invoke-SystemApi -PathValue ("/v1/system/skills?user_id={0}&include_inactive=true&limit=100" -f [uri]::EscapeDataString($UserId)) -Headers $headers
    $analyzer = Invoke-SystemApi -PathValue "/v1/system/analyzer?limit=20" -Headers $headers

    $snapshot = Build-Snapshot -Runtime $runtime -Status $status -Memory $memory -Skills $skills -Analyzer $analyzer
    $signature = Build-Signature -Snapshot $snapshot

    $shouldRecord = $RecordEveryPoll.IsPresent -or $signature -ne $lastSignature
    if ($shouldRecord) {
      ($snapshot | ConvertTo-Json -Depth 6 -Compress) | Add-Content -Path $jsonlPath
      $recorded += 1
      $lastSignature = $signature
    }

    if ($snapshot.health -ne "OK") {
      $healthDegradedCount += 1
    }

    Write-Line ("#{0} health={1} backend={2}/{3} mem={4} skills={5}/{6} fallback={7}" -f $iteration, $snapshot.health, $snapshot.backend_configured, $snapshot.backend_effective, $snapshot.memory_entries, $snapshot.skills_active, $snapshot.skills_total, $snapshot.fallback_rate)
  }
  catch {
    $errorCount += 1
    $err = [ordered]@{
      at_utc = (Get-Date).ToUniversalTime().ToString("o")
      health = "ERROR"
      error = $_.Exception.Message
    }
    ($err | ConvertTo-Json -Compress) | Add-Content -Path $jsonlPath
    $recorded += 1
    Write-Line ("#{0} error={1}" -f $iteration, $_.Exception.Message)
  }

  if ($RunOnce.IsPresent) {
    break
  }

  $nowUtc = (Get-Date).ToUniversalTime()
  if ($nowUtc -ge $deadline) {
    break
  }
  Start-Sleep -Seconds ($PollMinutes * 60)
}

$endUtc = (Get-Date).ToUniversalTime()
$summary = @(
  ("Full-Day Update Summary"),
  ("GeneratedAtUtc: {0}" -f $endUtc.ToString("o")),
  ("BaseUrl: {0}" -f $BaseUrl),
  ("UserId: {0}" -f $UserId),
  ("DurationHours: {0}" -f $DurationHours),
  ("PollMinutes: {0}" -f $PollMinutes),
  ("Iterations: {0}" -f $iteration),
  ("RecordedRows: {0}" -f $recorded),
  ("DegradedCount: {0}" -f $healthDegradedCount),
  ("ErrorCount: {0}" -f $errorCount),
  ("JsonlPath: {0}" -f $jsonlPath)
)
$summary | Set-Content -Path $summaryPath

Write-Line ("Completed. Summary: {0}" -f $summaryPath)
