param(
  [string]$BaseUrl = "https://unblemished-ai.onrender.com",
  [ValidateSet("chat_general", "chat_technical", "chat_companion", "status_runtime", "memory_skills")] [string]$Role = "chat_general",
  [string]$UserId = "team-train-user",
  [int]$DurationMinutes = 45,
  [int]$DelaySeconds = 10,
  [string]$AuthHeader = "",
  [string]$AuthValue = "",
  [string]$LogPath = "data/daily_updates/team_train_worker.jsonl",
  [string]$StopFlagPath = "",
  [switch]$RunOnce
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($DurationMinutes -lt 1) { throw "DurationMinutes must be >= 1" }
if ($DelaySeconds -lt 0) { throw "DelaySeconds must be >= 0" }

function Ensure-ParentDir {
  param([string]$PathValue)
  $parent = Split-Path -Parent $PathValue
  if ($parent -and -not (Test-Path $parent)) {
    New-Item -Path $parent -ItemType Directory | Out-Null
  }
}

function Build-Headers {
  $headers = @{}
  if ($AuthHeader -and $AuthValue) {
    $headers[$AuthHeader] = $AuthValue
  }
  return $headers
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

function Invoke-ApiGetSafe {
  param([string]$PathValue, [hashtable]$Headers)
  $uri = "{0}{1}" -f $BaseUrl.TrimEnd("/"), $PathValue
  try {
    $data = Invoke-RestMethod -Method GET -Uri $uri -Headers $Headers -TimeoutSec 25
    return @{ ok = $true; data = $data; error = "" }
  }
  catch {
    return @{ ok = $false; data = $null; error = $_.Exception.Message }
  }
}

function Invoke-ApiPostSafe {
  param([string]$PathValue, [hashtable]$Headers, [string]$BodyJson)
  $uri = "{0}{1}" -f $BaseUrl.TrimEnd("/"), $PathValue
  try {
    $data = Invoke-RestMethod -Method POST -Uri $uri -Headers $Headers -ContentType "application/json" -Body $BodyJson -TimeoutSec 35
    return @{ ok = $true; data = $data; error = "" }
  }
  catch {
    return @{ ok = $false; data = $null; error = $_.Exception.Message }
  }
}

function Write-Event {
  param([hashtable]$Event)
  ($Event | ConvertTo-Json -Compress) | Add-Content -Path $LogPath
}

function Write-Info {
  param([string]$Message)
  Write-Host ("[{0}] [{1}] {2}" -f (Get-Date).ToString("HH:mm:ss"), $Role, $Message)
}

function Get-PromptPool {
  param([string]$RoleName)
  if ($RoleName -eq "chat_general") {
    return @(
      "What is AI in one clear line?",
      "Explain machine learning in easy words.",
      "How can I improve coding daily in 1 hour?",
      "Define vector database in short.",
      "Summarize why memory retrieval matters."
    )
  }
  if ($RoleName -eq "chat_technical") {
    return @(
      "Design retry strategy for payment gateway failures.",
      "Refine with exponential backoff and jitter.",
      "Explain circuit breaker and timeout budget.",
      "How to reduce fallback rate in orchestrator?",
      "Give production deployment checklist for this app."
    )
  }
  if ($RoleName -eq "chat_companion") {
    return @(
      "Hello, how are you today?",
      "Talk in warm friendly style and ask one useful follow-up.",
      "Motivate me with one practical step for today.",
      "Help me calm down before work in 2 lines.",
      "Give one gentle reminder for focus."
    )
  }
  return @()
}

function Test-ChatAnswerQuality {
  param([string]$Answer)
  $reasons = @()
  $text = ($Answer -replace "\s+", " ").Trim()
  if (-not $text) { $reasons += "empty_answer" }
  if ($text.Length -lt 24) { $reasons += "too_short" }
  if ($text -match "(?i)routed model:|safe fallback|direct answer for your query|failed after retries") {
    $reasons += "template_or_fallback"
  }
  if ($text -match "(?i)unable to|internal server error|error") {
    $reasons += "error_like"
  }

  $pass = ($reasons.Count -eq 0)
  $reason = "ok"
  if (-not $pass) { $reason = ($reasons -join ",") }
  return @{ pass = $pass; reason = $reason }
}

function Compute-MemoryStatusPercent {
  param([object]$StatusObj, [object]$MemoryObj)
  $readyValue = Get-Optional -ObjectValue $MemoryObj -PropertyName "memory_ready" -DefaultValue (Get-Optional -ObjectValue $StatusObj -PropertyName "memory_ready" -DefaultValue $false)
  $ready = ($readyValue -eq $true)
  $rawEntries = [int](Get-Optional -ObjectValue $MemoryObj -PropertyName "memory_entries_count" -DefaultValue (Get-Optional -ObjectValue $StatusObj -PropertyName "memory_entries_count" -DefaultValue 0))
  $entries = if ($rawEntries -gt 0) { $rawEntries } else { 0 }
  $embeddingEnabled = (Get-Optional -ObjectValue $MemoryObj -PropertyName "embedding_enabled" -DefaultValue $true) -eq $true

  $score = 0
  if ($ready) { $score += 70 }
  $score += [Math]::Min($entries, 30)
  if (-not $embeddingEnabled) {
    $score = [Math]::Min($score, 70)
  }
  return [Math]::Max(0, [Math]::Min(100, [int][Math]::Round($score)))
}

Ensure-ParentDir -PathValue $LogPath
if ($StopFlagPath) {
  Ensure-ParentDir -PathValue $StopFlagPath
}

$headers = Build-Headers
$deadline = (Get-Date).ToUniversalTime().AddMinutes($DurationMinutes)
$promptPool = Get-PromptPool -RoleName $Role
$promptCursor = 0
$degradedStreak = 0
$totalRounds = 0
$totalPass = 0

Write-Info ("Worker start. BaseUrl={0}, DurationMinutes={1}, LogPath={2}" -f $BaseUrl, $DurationMinutes, $LogPath)

while ((Get-Date).ToUniversalTime() -lt $deadline) {
  if ($StopFlagPath -and (Test-Path $StopFlagPath)) {
    Write-Info "Stop flag detected. Worker exiting."
    break
  }

  $totalRounds += 1
  $event = [ordered]@{
    at_utc = (Get-Date).ToUniversalTime().ToString("o")
    role = $Role
    round = $totalRounds
    ok = $false
  }

  if ($Role -eq "status_runtime") {
    $statusResult = Invoke-ApiGetSafe -PathValue "/v1/system/status" -Headers $headers
    $runtimeResult = Invoke-ApiGetSafe -PathValue "/v1/system/runtime" -Headers $headers
    $event.status_ok = $statusResult.ok
    $event.runtime_ok = $runtimeResult.ok

    if ($statusResult.ok -and $runtimeResult.ok) {
      $statusData = $statusResult.data
      $runtimeData = $runtimeResult.data
      $statusVal = [string](Get-Optional -ObjectValue $statusData -PropertyName "status" -DefaultValue "unknown")
      $backend = [string](Get-Optional -ObjectValue $runtimeData -PropertyName "model_backend_effective" -DefaultValue "unknown")
      $event.ok = $true
      $event.status = $statusVal
      $event.backend = $backend
      $event.memory_ready = (Get-Optional -ObjectValue $statusData -PropertyName "memory_ready" -DefaultValue $false)
      $event.skill_ready = (Get-Optional -ObjectValue $statusData -PropertyName "skill_engine_ready" -DefaultValue $false)

      if ($statusVal -ne "ok") {
        $degradedStreak += 1
      } else {
        $degradedStreak = 0
      }
      $event.degraded_streak = $degradedStreak

      if ($degradedStreak -ge 3 -and $StopFlagPath) {
        Set-Content -Path $StopFlagPath -Value ("degraded_at={0}" -f (Get-Date).ToUniversalTime().ToString("o"))
        $event.stop_triggered = "degraded_streak"
        Write-Info "Degraded streak >= 3. Stop flag created."
      }
    } else {
      $event.error = "{0} | {1}" -f $statusResult.error, $runtimeResult.error
    }
    Write-Event -Event $event
    Write-Info ("round={0} ok={1} degraded_streak={2}" -f $totalRounds, $event.ok, (Get-Optional -ObjectValue $event -PropertyName "degraded_streak" -DefaultValue 0))
  }
  elseif ($Role -eq "memory_skills") {
    $statusResult = Invoke-ApiGetSafe -PathValue "/v1/system/status" -Headers $headers
    $memoryResult = Invoke-ApiGetSafe -PathValue "/v1/system/memory" -Headers $headers
    $skillsResult = Invoke-ApiGetSafe -PathValue ("/v1/system/skills?user_id={0}&include_inactive=true&limit=100" -f [uri]::EscapeDataString($UserId)) -Headers $headers

    if ($statusResult.ok -and $memoryResult.ok -and $skillsResult.ok) {
      $statusData = $statusResult.data
      $memoryData = $memoryResult.data
      $skillsData = $skillsResult.data
      $memoryPct = Compute-MemoryStatusPercent -StatusObj $statusData -MemoryObj $memoryData
      $event.ok = $true
      $event.memory_ready = (Get-Optional -ObjectValue $memoryData -PropertyName "memory_ready" -DefaultValue $false)
      $event.memory_entries = (Get-Optional -ObjectValue $memoryData -PropertyName "memory_entries_count" -DefaultValue 0)
      $event.memory_status_pct = $memoryPct
      $event.skills_total = (Get-Optional -ObjectValue $skillsData -PropertyName "total_count" -DefaultValue 0)
      $event.skills_active = (Get-Optional -ObjectValue $skillsData -PropertyName "active_count" -DefaultValue 0)
      Write-Info ("round={0} memory_pct={1}% skills={2}/{3}" -f $totalRounds, $memoryPct, $event.skills_active, $event.skills_total)
    } else {
      $event.error = "{0} | {1} | {2}" -f $statusResult.error, $memoryResult.error, $skillsResult.error
      Write-Info ("round={0} error={1}" -f $totalRounds, $event.error)
    }
    Write-Event -Event $event
  }
  else {
    if ($promptPool.Count -eq 0) {
      $event.error = "prompt_pool_empty"
      Write-Event -Event $event
      break
    }

    if ($promptCursor -ge $promptPool.Count) {
      $promptCursor = 0
      $promptPool = $promptPool | Sort-Object { Get-Random }
    }
    $prompt = [string]$promptPool[$promptCursor]
    $promptCursor += 1

    $path = "/v1/chat/reason"
    $answerStyle = "factual"
    if ($Role -eq "chat_companion") {
      $path = "/v1/chat/companion"
      $answerStyle = "relational"
    }
    if ($Role -eq "chat_technical") {
      $answerStyle = "technical"
    }

    $bodyObj = @{
      input_text = $prompt
      user_id = ("{0}-{1}" -f $UserId, $Role)
      context = @{
        source = "team-train-5-routes"
        role = $Role
        answer_style = $answerStyle
      }
    }
    $bodyJson = $bodyObj | ConvertTo-Json -Depth 8 -Compress
    $postResult = Invoke-ApiPostSafe -PathValue $path -Headers $headers -BodyJson $bodyJson
    $event.prompt = $prompt
    $event.path = $path

    if ($postResult.ok) {
      $answer = [string](Get-Optional -ObjectValue $postResult.data -PropertyName "final_answer" -DefaultValue "")
      $quality = Test-ChatAnswerQuality -Answer $answer
      $event.ok = $true
      $event.quality_pass = $quality.pass
      $event.quality_reason = $quality.reason
      $event.answer_excerpt = if ($answer.Length -le 240) { $answer } else { $answer.Substring(0, 240) }
      if ($quality.pass -eq $true) { $totalPass += 1 }
      Write-Info ("round={0} pass={1} reason={2}" -f $totalRounds, $quality.pass, $quality.reason)
    } else {
      $event.error = $postResult.error
      Write-Info ("round={0} error={1}" -f $totalRounds, $postResult.error)
    }
    Write-Event -Event $event
  }

  if ($RunOnce.IsPresent) { break }
  if ($DelaySeconds -gt 0) { Start-Sleep -Seconds $DelaySeconds }
}

$passRate = 0.0
if ($totalRounds -gt 0) {
  $passRate = [math]::Round(($totalPass / $totalRounds), 4)
}
$final = [ordered]@{
  at_utc = (Get-Date).ToUniversalTime().ToString("o")
  role = $Role
  event = "worker_summary"
  rounds = $totalRounds
  total_pass = $totalPass
  pass_rate = $passRate
}
Write-Event -Event $final
Write-Info ("Worker done. rounds={0}, pass_rate={1}" -f $totalRounds, $passRate)
