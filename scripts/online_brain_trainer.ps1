param(
  [string]$BaseUrl = "https://unblemished-ai.onrender.com",
  [string]$UserId = "online-train-user",
  [int]$MaxRounds = 200,
  [int]$DelaySeconds = 2,
  [string]$ContextJson = '{"source":"online-trainer","answer_style":"factual","priority":"high"}',
  [string]$PromptFile = "",
  [string]$AuthHeader = "",
  [string]$AuthValue = "",
  [int]$MaxConsecutiveErrors = 5,
  [string]$OutDir = "data/daily_updates",
  [switch]$StopOnDegraded = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($MaxRounds -lt 1) {
  throw "MaxRounds must be >= 1"
}
if ($DelaySeconds -lt 0) {
  throw "DelaySeconds must be >= 0"
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

function Invoke-JsonGet {
  param([string]$PathValue, [hashtable]$Headers)
  $uri = "{0}{1}" -f $BaseUrl.TrimEnd("/"), $PathValue
  return Invoke-RestMethod -Method GET -Uri $uri -Headers $Headers -TimeoutSec 25
}

function Invoke-JsonPost {
  param([string]$PathValue, [hashtable]$Headers, [string]$BodyJson)
  $uri = "{0}{1}" -f $BaseUrl.TrimEnd("/"), $PathValue
  return Invoke-RestMethod -Method POST -Uri $uri -Headers $Headers -ContentType "application/json" -Body $BodyJson -TimeoutSec 35
}

function Get-Optional {
  param([object]$ObjectValue, [string]$PropertyName, [object]$DefaultValue = $null)
  if ($null -eq $ObjectValue) { return $DefaultValue }
  $prop = $ObjectValue.PSObject.Properties[$PropertyName]
  if ($null -eq $prop -or $null -eq $prop.Value) { return $DefaultValue }
  return $prop.Value
}

function Normalize-Answer {
  param([string]$Text)
  if (-not $Text) { return "" }
  $line = ($Text -replace "\s+", " ").Trim()
  if ($line.Length -le 260) { return $line }
  return $line.Substring(0, 260)
}

function Load-Prompts {
  param([string]$PathValue)
  if ($PathValue -and (Test-Path $PathValue)) {
    $lines = Get-Content $PathValue | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" -and -not $_.StartsWith("#") }
    if ($lines.Count -gt 0) {
      return ,$lines
    }
  }
  $builtIn = @(
    "Define AI in one clear line.",
    "Explain vector database in short.",
    "How to design retry strategy for payment failures?",
    "Refine it with exponential backoff and jitter.",
    "What is circuit breaker in API systems?",
    "How to reduce fallback rate in orchestrator?",
    "Design schema for profile goal memory tables.",
    "How to detect prompt injection in tool calls?",
    "Give rate-limit policy for multi-user chat API.",
    "How to improve observability for this app?",
    "Why memory retrieval quality gets weak over time?",
    "Give concise incident checklist for backend outage.",
    "Summarize user preference extraction strategy.",
    "How to keep responses short but useful?",
    "What should be tested before production release?",
    "How to troubleshoot skill trigger mismatch?",
    "Give step-by-step plan for safe deploy.",
    "Explain semantic embeddings in easy words.",
    "How to tune top-k memory retrieval?",
    "What is the difference between heuristic and LLM?"
  )
  return ,$builtIn
}

function ConvertTo-HashtableCompat {
  param([object]$InputObject)
  if ($null -eq $InputObject) {
    return @{}
  }
  if ($InputObject -is [hashtable]) {
    return $InputObject
  }
  if ($InputObject -is [System.Collections.IDictionary]) {
    $output = @{}
    foreach ($key in $InputObject.Keys) {
      $output[[string]$key] = ConvertTo-HashtableCompat -InputObject $InputObject[$key]
    }
    return $output
  }
  if ($InputObject -is [System.Collections.IEnumerable] -and -not ($InputObject -is [string])) {
    $arrayOutput = @()
    foreach ($item in $InputObject) {
      $arrayOutput += ,(ConvertTo-HashtableCompat -InputObject $item)
    }
    return $arrayOutput
  }
  if ($InputObject.PSObject -and $null -ne $InputObject.PSObject.Properties) {
    $output = @{}
    foreach ($prop in $InputObject.PSObject.Properties) {
      $output[$prop.Name] = ConvertTo-HashtableCompat -InputObject $prop.Value
    }
    return $output
  }
  return $InputObject
}

function Write-Info {
  param([string]$Message)
  Write-Host ("[{0}] {1}" -f (Get-Date).ToString("HH:mm:ss"), $Message)
}

Ensure-Directory -PathValue $OutDir
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$jsonlPath = Join-Path $OutDir ("online_brain_train_{0}.jsonl" -f $stamp)
$summaryPath = Join-Path $OutDir ("online_brain_train_summary_{0}.txt" -f $stamp)

$headers = Build-Headers
$contextParsed = $ContextJson | ConvertFrom-Json
$contextObj = ConvertTo-HashtableCompat -InputObject $contextParsed
$prompts = Load-Prompts -PathValue $PromptFile
if ($prompts.Count -lt 1) {
  throw "No prompts available."
}

$successCount = 0
$degradedCount = 0
$errorCount = 0
$consecutiveErrors = 0
$stopReason = "max_rounds_reached"
$startUtc = (Get-Date).ToUniversalTime().ToString("o")

Write-Info ("Online trainer started. BaseUrl={0}, UserId={1}, MaxRounds={2}, PromptPool={3}" -f $BaseUrl, $UserId, $MaxRounds, $prompts.Count)
Write-Info ("Output JSONL: {0}" -f $jsonlPath)

$cursor = 0
for ($round = 1; $round -le $MaxRounds; $round++) {
  $prompt = [string]$prompts[$cursor]
  $cursor++
  if ($cursor -ge $prompts.Count) {
    $cursor = 0
    $prompts = $prompts | Sort-Object { Get-Random }
  }

  try {
    $status = Invoke-JsonGet -PathValue "/v1/system/status" -Headers $headers
    $runtime = Invoke-JsonGet -PathValue "/v1/system/runtime" -Headers $headers
    $health = [string](Get-Optional -ObjectValue $status -PropertyName "status" -DefaultValue "unknown")
    if ($health -ne "ok") {
      $degradedCount += 1
      if ($StopOnDegraded.IsPresent) {
        $stopReason = "server_degraded"
        $event = [ordered]@{
          at_utc = (Get-Date).ToUniversalTime().ToString("o")
          round = $round
          event = "stop"
          reason = $stopReason
          status = $health
          backend = (Get-Optional -ObjectValue $runtime -PropertyName "model_backend_effective" -DefaultValue "unknown")
        }
        ($event | ConvertTo-Json -Compress) | Add-Content -Path $jsonlPath
        Write-Info ("Stop on degraded health at round={0}" -f $round)
        break
      }
    }

    $bodyObj = @{
      input_text = $prompt
      user_id = $UserId
      context = $contextObj
    }
    $bodyJson = $bodyObj | ConvertTo-Json -Depth 8 -Compress

    $started = Get-Date
    $reply = Invoke-JsonPost -PathValue "/v1/chat/reason" -Headers $headers -BodyJson $bodyJson
    $latencyMs = [int]((Get-Date) - $started).TotalMilliseconds
    $finalAnswer = [string](Get-Optional -ObjectValue $reply -PropertyName "final_answer" -DefaultValue "")

    $telemetry = Get-Optional -ObjectValue $status -PropertyName "telemetry" -DefaultValue $null
    $eventOk = [ordered]@{
      at_utc = (Get-Date).ToUniversalTime().ToString("o")
      round = $round
      prompt = $prompt
      answer_excerpt = (Normalize-Answer -Text $finalAnswer)
      latency_ms = $latencyMs
      status = $health
      backend = (Get-Optional -ObjectValue $runtime -PropertyName "model_backend_effective" -DefaultValue "unknown")
      model_ready = (Get-Optional -ObjectValue $status -PropertyName "model_ready" -DefaultValue $false)
      memory_ready = (Get-Optional -ObjectValue $status -PropertyName "memory_ready" -DefaultValue $false)
      skill_ready = (Get-Optional -ObjectValue $status -PropertyName "skill_engine_ready" -DefaultValue $false)
      fallback_rate = (Get-Optional -ObjectValue $telemetry -PropertyName "fallback_rate" -DefaultValue 0)
      request_count = (Get-Optional -ObjectValue $telemetry -PropertyName "request_count" -DefaultValue 0)
    }
    ($eventOk | ConvertTo-Json -Compress) | Add-Content -Path $jsonlPath

    $successCount += 1
    $consecutiveErrors = 0
    Write-Info ("#{0} ok backend={1} latency={2}ms" -f $round, $eventOk.backend, $latencyMs)
  }
  catch {
    $errorCount += 1
    $consecutiveErrors += 1
    $errEvent = [ordered]@{
      at_utc = (Get-Date).ToUniversalTime().ToString("o")
      round = $round
      prompt = $prompt
      event = "error"
      message = $_.Exception.Message
      consecutive_errors = $consecutiveErrors
    }
    ($errEvent | ConvertTo-Json -Compress) | Add-Content -Path $jsonlPath
    Write-Info ("#{0} error={1}" -f $round, $_.Exception.Message)

    if ($consecutiveErrors -ge $MaxConsecutiveErrors) {
      $stopReason = "max_consecutive_errors"
      Write-Info ("Stopping due to consecutive errors={0}" -f $consecutiveErrors)
      break
    }
  }

  if ($round -ge $MaxRounds) {
    break
  }
  if ($DelaySeconds -gt 0) {
    Start-Sleep -Seconds $DelaySeconds
  }
}

$endUtc = (Get-Date).ToUniversalTime().ToString("o")
$summaryLines = @(
  "Online Brain Trainer Summary",
  ("StartedAtUtc: {0}" -f $startUtc),
  ("EndedAtUtc: {0}" -f $endUtc),
  ("BaseUrl: {0}" -f $BaseUrl),
  ("UserId: {0}" -f $UserId),
  ("MaxRounds: {0}" -f $MaxRounds),
  ("SuccessCount: {0}" -f $successCount),
  ("DegradedCount: {0}" -f $degradedCount),
  ("ErrorCount: {0}" -f $errorCount),
  ("StopReason: {0}" -f $stopReason),
  ("JsonlPath: {0}" -f $jsonlPath)
)
$summaryLines | Set-Content -Path $summaryPath
Write-Info ("Completed. Summary: {0}" -f $summaryPath)
