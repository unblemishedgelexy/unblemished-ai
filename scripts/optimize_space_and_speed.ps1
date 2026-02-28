param(
  [switch]$CreateRuntimeZip,
  [string]$RuntimeZipPath = "dist/humoniod_runtime_compact.zip",
  [switch]$StartFastServer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Remove-IfExists {
  param([string]$PathValue)
  if (Test-Path $PathValue) {
    Remove-Item $PathValue -Recurse -Force
    return $true
  }
  return $false
}

Write-Host "=== Humoniod Optimize: Space + Speed ===" -ForegroundColor Cyan

$removedFiles = 0
$removedDirs = 0

$filePatterns = @(
  "data\*.log",
  "data\live_dummy_chat_*_latest.jsonl",
  "data\flirty_*_attempt*.json",
  "data\terminal_chat_room.log"
)

foreach ($pattern in $filePatterns) {
  $matches = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue
  foreach ($item in $matches) {
    Remove-Item $item.FullName -Force
    $removedFiles++
  }
}

$cacheDirs = @()
$cacheDirs += Get-ChildItem app -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
$cacheDirs += Get-ChildItem scripts -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue

if (Test-Path ".pytest_cache") {
  $cacheDirs += Get-Item ".pytest_cache"
}

foreach ($dir in $cacheDirs) {
  if (Remove-IfExists -PathValue $dir.FullName) {
    $removedDirs++
  }
}

Write-Host ("Removed files : {0}" -f $removedFiles) -ForegroundColor Green
Write-Host ("Removed dirs  : {0}" -f $removedDirs) -ForegroundColor Green

if ($CreateRuntimeZip) {
  $zipFullPath = Join-Path $repoRoot $RuntimeZipPath
  $zipDir = Split-Path -Parent $zipFullPath
  if ($zipDir -and -not (Test-Path $zipDir)) {
    New-Item -ItemType Directory -Path $zipDir | Out-Null
  }

  $stageRoot = Join-Path $env:TEMP ("humoniod_compact_" + [guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Path $stageRoot | Out-Null

  $includeItems = @(
    "app",
    "scripts",
    "README.md",
    "requirements.txt"
  )

  foreach ($item in $includeItems) {
    if (Test-Path $item) {
      Copy-Item -Path $item -Destination $stageRoot -Recurse -Force
    }
  }

  $stageTestsPath = Join-Path $stageRoot "app\tests"
  if (Test-Path $stageTestsPath) {
    Remove-Item $stageTestsPath -Recurse -Force
  }

  Get-ChildItem $stageRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force }

  if (Test-Path $zipFullPath) {
    Remove-Item $zipFullPath -Force
  }

  Compress-Archive -Path (Join-Path $stageRoot "*") -DestinationPath $zipFullPath -CompressionLevel Optimal
  Remove-Item $stageRoot -Recurse -Force
  Write-Host ("Created runtime zip: {0}" -f $zipFullPath) -ForegroundColor Green
}

if ($StartFastServer) {
  Write-Host "Applying fast runtime profile..." -ForegroundColor Cyan

  if (-not $env:MODEL_BACKEND) { $env:MODEL_BACKEND = "local_llama" }
  if (-not $env:REASONING_MODE) { $env:REASONING_MODE = "fast" }

  $env:STRICT_RESPONSE_MODE = "false"
  $env:SELF_EVALUATION_ENABLED = "false"
  $env:RESPONSE_MATCH_MODEL_ENABLED = "false"
  $env:MODEL_MAX_RETRIES = "0"
  $env:MEMORY_TOP_K = "2"
  $env:MEMORY_CONTEXT_MAX_TOKENS = "120"
  $env:MEMORY_LONG_TERM_EVERY_N_MESSAGES = "12"
  $env:PRIVACY_REDACTION_ENABLED = "true"
  $env:PRIVACY_REMOVE_SENSITIVE_CONTEXT_KEYS = "true"
  $env:RELATIONSHIP_MEMORY_TEXT_ENABLED = "false"

  Remove-Item Env:OPENAI_COMPAT_BASE_URL -ErrorAction SilentlyContinue
  Remove-Item Env:OPENAI_COMPAT_API_KEY -ErrorAction SilentlyContinue
  Remove-Item Env:OPENAI_COMPAT_MODEL -ErrorAction SilentlyContinue

  if ($env:MODEL_BACKEND -eq "local_llama") {
    if (-not $env:LOCAL_MODEL_PATH) {
      Write-Host "Warning: LOCAL_MODEL_PATH empty. Set it before llama start." -ForegroundColor Yellow
    }
  }

  & ".\.venv\Scripts\python" -m uvicorn app.main:app --reload
}
