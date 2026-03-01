param(
  [string]$BaseUrl = "https://unblemished-ai.onrender.com",
  [string]$UserId = "online-train-user",
  [int]$MaxRounds = 300,
  [int]$DelaySeconds = 2,
  [string]$ContextJson = '{"source":"online-trainer","answer_style":"factual","priority":"high"}',
  [string]$PromptFile = "",
  [string]$AuthHeader = "",
  [string]$AuthValue = "",
  [int]$MaxConsecutiveErrors = 5,
  [double]$TargetPassRate = 0.80,
  [int]$RollingWindow = 20,
  [int]$MinRoundsPerPool = 20,
  [string]$OutDir = "data/daily_updates",
  [switch]$StopOnDegraded = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($MaxRounds -lt 1) { throw "MaxRounds must be >= 1" }
if ($DelaySeconds -lt 0) { throw "DelaySeconds must be >= 0" }
if ($RollingWindow -lt 5) { throw "RollingWindow must be >= 5" }
if ($MinRoundsPerPool -lt 5) { throw "MinRoundsPerPool must be >= 5" }
if ($TargetPassRate -lt 0.1 -or $TargetPassRate -gt 1.0) { throw "TargetPassRate must be in [0.1, 1.0]" }

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

function ConvertTo-HashtableCompat {
  param([object]$InputObject)
  if ($null -eq $InputObject) { return @{} }
  if ($InputObject -is [hashtable]) { return $InputObject }
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

function New-PromptPool {
  param([string]$Name, [string[]]$Prompts)
  return @{
    name = $Name
    prompts = @($Prompts)
  }
}

function Load-PromptPools {
  param([string]$PathValue)
  if ($PathValue -and (Test-Path $PathValue)) {
    $lines = Get-Content $PathValue | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" -and -not $_.StartsWith("#") }
    if ($lines.Count -gt 0) {
      return ,(New-PromptPool -Name "custom" -Prompts $lines)
    }
  }

  $poolA = @(
    "What is AI in one line?",
    "Define machine learning in simple words.",
    "Explain vector database briefly.",
    "What is an API?",
    "Difference between heuristic and LLM?"
  )
  $poolB = @(
    "Design retry strategy for payment gateway failures.",
    "Refine strategy with exponential backoff and jitter.",
    "Explain circuit breaker with timeout budget.",
    "How to reduce fallback rate in orchestrator?",
    "Give a practical observability checklist for this app."
  )
  $poolC = @(
    "Troubleshoot why skill trigger does not run.",
    "Why memory quality may degrade over time?",
    "How to validate response quality before returning?",
    "Give production-safe rollout plan in steps.",
    "How to detect prompt injection in tool calls?"
  )

  return @(
    (New-PromptPool -Name "baseline" -Prompts $poolA),
    (New-PromptPool -Name "technical" -Prompts $poolB),
    (New-PromptPool -Name "troubleshoot" -Prompts $poolC)
  )
}

function Evaluate-AnswerQuality {
  param(
    [string]$Prompt,
    [string]$Answer
  )

  $reasons = @()
  $answerNorm = ($Answer -replace "\s+", " ").Trim()
  $promptNorm = ($Prompt -replace "\s+", " ").Trim().ToLowerInvariant()

  if (-not $answerNorm) {
    $reasons += "empty_answer"
  }
  if ($answerNorm.Length -lt 24) {
    $reasons += "too_short"
  }
  if ($answerNorm -match "(?i)routed model:|safe fallback|direct answer for your query") {
    $reasons += "template_or_fallback"
  }
  if ($answerNorm -match "(?i)unable to|error|failed after retries") {
    $reasons += "error_like"
  }

  if ($promptNorm -match "what is ai|define machine learning|vector database|api") {
    if ($answerNorm -notmatch "(?i)ai|intelligence|learning|vector|database|api") {
      $reasons += "missing_core_term"
    }
  }

  $pass = ($reasons.Count -eq 0)
  $reasonText = "ok"
  if (-not $pass) {
    $reasonText = ($reasons -join ",")
  }
  return @{
    pass = $pass
    reason = $reasonText
  }
}

function Get-RollingPassRate {
  param([object[]]$Window)
  if ($Window.Count -eq 0) { return 0.0 }
  $hits = 0
  foreach ($item in $Window) {
    if ($item -eq $true) { $hits += 1 }
  }
  return [math]::Round(($hits / $Window.Count), 4)
}

Ensure-Directory -PathValue $OutDir
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$jsonlPath = Join-Path $OutDir ("online_brain_train_{0}.jsonl" -f $stamp)
$summaryPath = Join-Path $OutDir ("online_brain_train_summary_{0}.txt" -f $stamp)

$headers = Build-Headers
$contextParsed = $ContextJson | ConvertFrom-Json
$contextObj = ConvertTo-HashtableCompat -InputObject $contextParsed
$promptPools = Load-PromptPools -PathValue $PromptFile
if ($promptPools.Count -lt 1) {
  throw "No prompt pools available."
}

$successCount = 0
$qualityPassCount = 0
$degradedCount = 0
$errorCount = 0
$consecutiveErrors = 0
$stopReason = "max_rounds_reached"
$startUtc = (Get-Date).ToUniversalTime().ToString("o")

$poolIndex = 0
$poolRound = 0
$poolCursor = 0
$rolling = @()
$poolSwitchCount = 0

Write-Info ("Online trainer started. BaseUrl={0}, UserId={1}, MaxRounds={2}, Pools={3}" -f $BaseUrl, $UserId, $MaxRounds, $promptPools.Count)
Write-Info ("Output JSONL: {0}" -f $jsonlPath)

for ($round = 1; $round -le $MaxRounds; $round++) {
  $activePool = $promptPools[$poolIndex]
  $poolName = [string]$activePool.name
  $prompts = @($activePool.prompts)
  if ($prompts.Count -lt 1) { break }

  if ($poolCursor -ge $prompts.Count) {
    $poolCursor = 0
    $prompts = $prompts | Sort-Object { Get-Random }
    $activePool.prompts = $prompts
  }
  $prompt = [string]$prompts[$poolCursor]
  $poolCursor += 1
  $poolRound += 1

  try {
    $status = Invoke-JsonGet -PathValue "/v1/system/status" -Headers $headers
    $runtime = Invoke-JsonGet -PathValue "/v1/system/runtime" -Headers $headers
    $memory = Invoke-JsonGet -PathValue "/v1/system/memory" -Headers $headers
    $health = [string](Get-Optional -ObjectValue $status -PropertyName "status" -DefaultValue "unknown")
    if ($health -ne "ok") {
      $degradedCount += 1
      if ($StopOnDegraded.IsPresent) {
        $stopReason = "server_degraded"
        $event = [ordered]@{
          at_utc = (Get-Date).ToUniversalTime().ToString("o")
          round = $round
          pool = $poolName
          event = "stop"
          reason = $stopReason
          status = $health
          backend = (Get-Optional -ObjectValue $runtime -PropertyName "model_backend_effective" -DefaultValue "unknown")
          memory_ready = (Get-Optional -ObjectValue $status -PropertyName "memory_ready" -DefaultValue $false)
          skill_ready = (Get-Optional -ObjectValue $status -PropertyName "skill_engine_ready" -DefaultValue $false)
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
    $quality = Evaluate-AnswerQuality -Prompt $prompt -Answer $finalAnswer
    $qualityPass = [bool]$quality.pass
    if ($qualityPass) { $qualityPassCount += 1 }

    $rolling += ,$qualityPass
    while ($rolling.Count -gt $RollingWindow) {
      $rolling = @($rolling[1..($rolling.Count - 1)])
    }
    $rollingRate = Get-RollingPassRate -Window $rolling

    $telemetry = Get-Optional -ObjectValue $status -PropertyName "telemetry" -DefaultValue $null
    $eventOk = [ordered]@{
      at_utc = (Get-Date).ToUniversalTime().ToString("o")
      round = $round
      pool = $poolName
      pool_round = $poolRound
      prompt = $prompt
      answer_excerpt = (Normalize-Answer -Text $finalAnswer)
      latency_ms = $latencyMs
      status = $health
      backend = (Get-Optional -ObjectValue $runtime -PropertyName "model_backend_effective" -DefaultValue "unknown")
      model_ready = (Get-Optional -ObjectValue $status -PropertyName "model_ready" -DefaultValue $false)
      memory_ready = (Get-Optional -ObjectValue $status -PropertyName "memory_ready" -DefaultValue $false)
      skill_ready = (Get-Optional -ObjectValue $status -PropertyName "skill_engine_ready" -DefaultValue $false)
      memory_entries = (Get-Optional -ObjectValue $memory -PropertyName "memory_entries_count" -DefaultValue 0)
      quality_pass = $qualityPass
      quality_reason = [string]$quality.reason
      rolling_pass_rate = $rollingRate
      fallback_rate = (Get-Optional -ObjectValue $telemetry -PropertyName "fallback_rate" -DefaultValue 0)
      request_count = (Get-Optional -ObjectValue $telemetry -PropertyName "request_count" -DefaultValue 0)
    }
    ($eventOk | ConvertTo-Json -Compress) | Add-Content -Path $jsonlPath

    $successCount += 1
    $consecutiveErrors = 0
    Write-Info ("#{0} pool={1} pass={2} roll={3} backend={4}" -f $round, $poolName, $qualityPass, $rollingRate, $eventOk.backend)

    if ($poolRound -ge $MinRoundsPerPool -and $rollingRate -ge $TargetPassRate) {
      if ($poolIndex -lt ($promptPools.Count - 1)) {
        $prevPool = $poolName
        $poolIndex += 1
        $poolRound = 0
        $poolCursor = 0
        $rolling = @()
        $poolSwitchCount += 1
        $switchEvent = [ordered]@{
          at_utc = (Get-Date).ToUniversalTime().ToString("o")
          round = $round
          event = "pool_switch"
          from_pool = $prevPool
          to_pool = [string]$promptPools[$poolIndex].name
          reason = "rolling_pass_rate_reached"
          rolling_pass_rate = $rollingRate
          target_pass_rate = $TargetPassRate
        }
        ($switchEvent | ConvertTo-Json -Compress) | Add-Content -Path $jsonlPath
        Write-Info ("Switched pool {0} -> {1}" -f $switchEvent.from_pool, $switchEvent.to_pool)
      } else {
        $stopReason = "target_pass_rate_achieved_all_pools"
        Write-Info ("Target achieved across all pools. Stopping.")
        break
      }
    }
  }
  catch {
    $errorCount += 1
    $consecutiveErrors += 1
    $errEvent = [ordered]@{
      at_utc = (Get-Date).ToUniversalTime().ToString("o")
      round = $round
      pool = [string]$promptPools[$poolIndex].name
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

  if ($round -ge $MaxRounds) { break }
  if ($DelaySeconds -gt 0) { Start-Sleep -Seconds $DelaySeconds }
}

$endUtc = (Get-Date).ToUniversalTime().ToString("o")
$qualityPassRate = if ($successCount -gt 0) { [math]::Round($qualityPassCount / $successCount, 4) } else { 0.0 }
$summaryLines = @(
  "Online Brain Trainer Summary",
  ("StartedAtUtc: {0}" -f $startUtc),
  ("EndedAtUtc: {0}" -f $endUtc),
  ("BaseUrl: {0}" -f $BaseUrl),
  ("UserId: {0}" -f $UserId),
  ("MaxRounds: {0}" -f $MaxRounds),
  ("PromptPools: {0}" -f $promptPools.Count),
  ("PoolSwitchCount: {0}" -f $poolSwitchCount),
  ("SuccessCount: {0}" -f $successCount),
  ("QualityPassCount: {0}" -f $qualityPassCount),
  ("QualityPassRate: {0}" -f $qualityPassRate),
  ("DegradedCount: {0}" -f $degradedCount),
  ("ErrorCount: {0}" -f $errorCount),
  ("StopReason: {0}" -f $stopReason),
  ("TargetPassRate: {0}" -f $TargetPassRate),
  ("RollingWindow: {0}" -f $RollingWindow),
  ("MinRoundsPerPool: {0}" -f $MinRoundsPerPool),
  ("JsonlPath: {0}" -f $jsonlPath)
)
$summaryLines | Set-Content -Path $summaryPath
Write-Info ("Completed. Summary: {0}" -f $summaryPath)
