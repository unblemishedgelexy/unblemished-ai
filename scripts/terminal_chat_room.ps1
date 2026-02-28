param(
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [string]$UserId = "terminal-user-001",
  [string]$AnswerStyle = "flirty",
  [string]$Source = "terminal-chat-room",
  [int]$TimeoutSec = 180,
  [string]$LogPath = "",
  [string]$OnceMessage = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($LogPath) {
  $logDir = Split-Path -Parent $LogPath
  if ($logDir -and -not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
  }
}

function Write-ChatLog {
  param(
    [string]$Role,
    [string]$Text
  )
  if (-not $LogPath) {
    return
  }
  $safeText = ($Text -replace "`r", " " -replace "`n", " ").Trim()
  $line = "{0} | {1} | user={2} | style={3} | {4}" -f `
    (Get-Date).ToUniversalTime().ToString("o"), $Role, $script:UserId, $script:AnswerStyle, $safeText
  Add-Content -Path $LogPath -Value $line
}

function Test-Runtime {
  try {
    $runtime = Invoke-RestMethod -Method GET -Uri "$BaseUrl/v1/system/runtime" -TimeoutSec 15
    return $runtime
  }
  catch {
    return $null
  }
}

function Invoke-ChatReason {
  param(
    [string]$InputText
  )
  $body = @{
    input_text = $InputText
    user_id = $script:UserId
    context = @{
      answer_style = $script:AnswerStyle
      source = $Source
    }
  } | ConvertTo-Json -Depth 6 -Compress

  $response = Invoke-RestMethod `
    -Method POST `
    -Uri "$BaseUrl/v1/chat/reason" `
    -ContentType "application/json" `
    -Body $body `
    -TimeoutSec $TimeoutSec

  return [string]$response.final_answer
}

function Format-AssistantDisplay {
  param([string]$RawText)

  if (-not $RawText) {
    return $RawText
  }

  $text = $RawText
  if ($text -match "(?s)Answer:\s*(.+?)\s*Reasoning:") {
    $text = $matches[1].Trim()
  }

  $text = [regex]::Replace($text, "(?:^|\s)#[A-Za-z0-9_]+", "")
  $text = [regex]::Replace($text, "\s+", " ").Trim()
  return $text
}

function Show-Help {
  Write-Host ""
  Write-Host "Commands:" -ForegroundColor Cyan
  Write-Host "  /help                Show commands"
  Write-Host "  /exit                Exit chat room"
  Write-Host "  /status              Show runtime status"
  Write-Host "  /show                Show active user/style/base URL"
  Write-Host "  /user <id>           Change user id"
  Write-Host "  /style <name>        Change answer style (flirty/factual/technical/relational)"
  Write-Host "  /clear               Clear terminal"
  Write-Host ""
}

$runtime = Test-Runtime
if ($null -eq $runtime) {
  Write-Host "Server not reachable at $BaseUrl" -ForegroundColor Red
  Write-Host "Start backend first, then run this script again." -ForegroundColor Yellow
  exit 1
}

if ($OnceMessage) {
  try {
    $reply = Invoke-ChatReason -InputText $OnceMessage
    $displayReply = Format-AssistantDisplay -RawText $reply
    Write-Host "YOU > $OnceMessage" -ForegroundColor Yellow
    Write-Host "AI  > $displayReply" -ForegroundColor Green
    Write-ChatLog -Role "USER" -Text $OnceMessage
    Write-ChatLog -Role "AI" -Text $displayReply
    exit 0
  }
  catch {
    Write-Host ("Request failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
  }
}

Write-Host ""
Write-Host "=== Terminal Chat Room ===" -ForegroundColor Cyan
Write-Host ("Base URL   : {0}" -f $BaseUrl)
Write-Host ("User ID    : {0}" -f $UserId)
Write-Host ("Style      : {0}" -f $AnswerStyle)
Write-Host ("Backend    : {0}" -f $runtime.model_backend_effective)
Write-Host ("ModelReady : {0}" -f $runtime.model_ready)
Write-Host ("SelfEval   : {0}" -f $runtime.self_evaluation_enabled)
$logDisplay = if ($LogPath) { $LogPath } else { "<disabled>" }
Write-Host ("Log file   : {0}" -f $logDisplay)
if ($runtime.self_evaluation_enabled) {
  Write-Host "Warning: self_evaluation_enabled=true. Chat me fallback aa sakta hai." -ForegroundColor Yellow
}
Show-Help

while ($true) {
  $text = Read-Host "YOU"
  if ($null -eq $text) {
    continue
  }
  $text = $text.Trim()
  if (-not $text) {
    continue
  }

  if ($text -eq "/exit") {
    Write-Host "Chat room closed." -ForegroundColor Cyan
    break
  }
  if ($text -eq "/help") {
    Show-Help
    continue
  }
  if ($text -eq "/clear") {
    Clear-Host
    continue
  }
  if ($text -eq "/show") {
    Write-Host ("Base URL: {0}" -f $BaseUrl)
    Write-Host ("User ID : {0}" -f $UserId)
    Write-Host ("Style   : {0}" -f $AnswerStyle)
    continue
  }
  if ($text -eq "/status") {
    $r = Test-Runtime
    if ($null -eq $r) {
      Write-Host "Runtime unavailable." -ForegroundColor Red
    }
    else {
      Write-Host ("backend={0}, mode={1}, ready={2}, judge={3}" -f `
          $r.model_backend_effective, $r.reasoning_mode, $r.model_ready, $r.response_match_model_enabled) -ForegroundColor Green
    }
    continue
  }
  if ($text -like "/user *") {
    $newUser = $text.Substring(6).Trim()
    if ($newUser) {
      $script:UserId = $newUser
      Write-Host ("User changed to: {0}" -f $script:UserId) -ForegroundColor Cyan
    }
    continue
  }
  if ($text -like "/style *") {
    $newStyle = $text.Substring(7).Trim()
    if ($newStyle) {
      $script:AnswerStyle = $newStyle
      Write-Host ("Style changed to: {0}" -f $script:AnswerStyle) -ForegroundColor Cyan
    }
    continue
  }

  try {
    Write-ChatLog -Role "USER" -Text $text
    $rawReply = Invoke-ChatReason -InputText $text
    $displayReply = Format-AssistantDisplay -RawText $rawReply
    Write-Host ("AI > {0}" -f $displayReply) -ForegroundColor Green
    Write-ChatLog -Role "AI" -Text $displayReply
  }
  catch {
    $err = ("Request failed: {0}" -f $_.Exception.Message)
    Write-Host $err -ForegroundColor Red
    Write-ChatLog -Role "ERROR" -Text $err
  }
}
