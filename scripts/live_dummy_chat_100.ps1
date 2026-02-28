param(
  [string]$BaseUrl = "http://127.0.0.1:8003",
  [string]$UserId = "live-dummy-100",
  [int]$Count = 100,
  [string]$AnswerStyle = "flirty",
  [string]$Source = "live-terminal-run",
  [string]$OutputPath = "data/live_dummy_chat_100.jsonl",
  [int]$TimeoutSec = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Count -lt 1) {
  throw "Count must be >= 1."
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path $outputDir)) {
  New-Item -ItemType Directory -Path $outputDir | Out-Null
}

if (Test-Path $OutputPath) {
  Remove-Item $OutputPath -Force
}

function New-DummyPrompt {
  param([int]$Index)

  $openers = @(
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "sun na",
    "are you there",
    "hii"
  )
  $asks = @(
    "mujhe",
    "please mujhe",
    "thoda",
    "ek chhota sa",
    "quickly",
    "aaj",
    "abhi"
  )
  $intents = @(
    "cute good morning line do",
    "romantic one-liner bolo",
    "sweet compliment do",
    "playful tease karo",
    "motivation do pyare tone me",
    "mood uplift karo",
    "stress kam karne ka tip do",
    "good night message do",
    "aaj ka mini plan do",
    "focus karne ka 1 step do",
    "meri tarif me 1 line bolo",
    "friendly check-in karo",
    "soft reminder do pani pine ka",
    "confidence boost line do",
    "thoda smile karao",
    "short caring reply do",
    "flirty but respectful reply do",
    "cute question pucho",
    "positive vibe do",
    "ek sweet text bhejo"
  )
  $extras = @(
    "with 1 line",
    "2 lines max",
    "short reply",
    "hinglish me",
    "emoji ke saath",
    "without long text"
  )

  $prefix = Get-Random -InputObject $openers
  $ask = Get-Random -InputObject $asks
  $intent = Get-Random -InputObject $intents
  $extra = Get-Random -InputObject $extras
  return "$prefix $ask $intent $extra [$Index]"
}

function Has-FlirtyMarker {
  param([string]$Text)

  $lower = $Text.ToLowerInvariant()
  $markers = @(
    "hey cutie,",
    "aww sweet one,",
    "haan jaan,",
    "hey lovely,",
    "cutie",
    "sweet",
    "jaan",
    "lovely",
    "darling",
    "dear",
    "baby",
    "love",
    "cute",
    "😊",
    "😉",
    "❤️"
  )
  foreach ($m in $markers) {
    if ($lower.Contains($m)) {
      return $true
    }
  }
  return $false
}

Write-Host ""
Write-Host "=== Live Dummy Chat Runner ===" -ForegroundColor Cyan
Write-Host "BaseUrl     : $BaseUrl"
Write-Host "UserId      : $UserId"
Write-Host "Count       : $Count"
Write-Host "AnswerStyle : $AnswerStyle"
Write-Host "OutputPath  : $OutputPath"
Write-Host ""

try {
  $runtime = Invoke-RestMethod -Method GET "$BaseUrl/v1/system/runtime" -TimeoutSec 20
  Write-Host ("Runtime OK  : backend={0}, mode={1}, model_ready={2}" -f `
      $runtime.model_backend_effective, $runtime.reasoning_mode, $runtime.model_ready) -ForegroundColor Green
}
catch {
  Write-Host ("Runtime check failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
  throw
}

Write-Host ""
Write-Host "Starting live run..." -ForegroundColor Cyan
Write-Host ""

$okCount = 0
$failCount = 0

for ($i = 1; $i -le $Count; $i++) {
  $prompt = New-DummyPrompt -Index $i
  $requestBody = @{
    input_text = $prompt
    user_id = $UserId
    context = @{
      answer_style = $AnswerStyle
      source = $Source
    }
  } | ConvertTo-Json -Depth 6 -Compress

  $started = Get-Date
  $aiText = ""
  $ok = $false
  $errorMessage = ""

  try {
    $response = Invoke-RestMethod `
      -Method POST `
      -Uri "$BaseUrl/v1/chat/reason" `
      -ContentType "application/json" `
      -Body $requestBody `
      -TimeoutSec $TimeoutSec
    $aiText = [string]$response.final_answer
    $ok = Has-FlirtyMarker -Text $aiText
  }
  catch {
    $errorMessage = $_.Exception.Message
    $aiText = "ERROR: $errorMessage"
    $ok = $false
  }

  $latencyMs = [int][math]::Round(((Get-Date) - $started).TotalMilliseconds)

  if ($ok) {
    $okCount++
  }
  else {
    $failCount++
  }

  Write-Host ("[{0}/{1}] USER: {2}" -f $i, $Count, $prompt) -ForegroundColor Yellow
  if ($ok) {
    Write-Host ("[{0}/{1}] AI  : {2}" -f $i, $Count, $aiText) -ForegroundColor Green
  }
  else {
    Write-Host ("[{0}/{1}] AI  : {2}" -f $i, $Count, $aiText) -ForegroundColor Red
  }
  Write-Host ("[{0}/{1}] ms  : {2}" -f $i, $Count, $latencyMs)
  Write-Host ""

  $line = [ordered]@{
    idx = $i
    at_utc = (Get-Date).ToUniversalTime().ToString("o")
    user_id = $UserId
    prompt = $prompt
    response = $aiText
    latency_ms = $latencyMs
    flirty_ok = $ok
    error = $errorMessage
  } | ConvertTo-Json -Compress

  Add-Content -Path $OutputPath -Value $line
}

Write-Host "=== Live Run Complete ===" -ForegroundColor Cyan
Write-Host ("Passed (flirty marker) : {0}" -f $okCount) -ForegroundColor Green
Write-Host ("Failed                 : {0}" -f $failCount) -ForegroundColor Red
Write-Host ("Saved log              : {0}" -f $OutputPath)
