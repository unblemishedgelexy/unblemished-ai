param(
  [string]$BindAddress = "127.0.0.1",
  [int]$Port = 8000,
  [string]$OnlineFrom = "09:00",
  [string]$OnlineTo = "23:30",
  [ValidateSet("heuristic", "local_light")] [string]$OfflineMode = "heuristic",
  [string]$OnlineModelPath = "D:\kanchana-ai\ai-model\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf",
  [string]$LightModelPath = "D:\kanchana-ai\ai-model\models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
  [int]$PollSeconds = 15,
  [int]$WarmupSeconds = 45,
  [string]$StatePath = "data/server_autopilot_state.json",
  [string]$LogPath = "data/server_autopilot.log",
  [switch]$RunOnce
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Ensure-ParentDir {
  param([string]$PathValue)
  $parent = Split-Path -Parent $PathValue
  if ($parent -and -not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
  }
}

Ensure-ParentDir -PathValue $StatePath
Ensure-ParentDir -PathValue $LogPath

function Write-AutopilotLog {
  param(
    [string]$Message,
    [string]$Level = "INFO"
  )
  $line = "{0} [{1}] {2}" -f (Get-Date).ToUniversalTime().ToString("o"), $Level, $Message
  Add-Content -Path $LogPath -Value $line
  Write-Host $line
}

function Test-Health {
  try {
    $health = Invoke-RestMethod -Method GET -Uri ("http://{0}:{1}/health" -f $BindAddress, $Port) -TimeoutSec 5
    return ($health.status -eq "ok")
  }
  catch {
    return $false
  }
}

function Load-State {
  if (-not (Test-Path $StatePath)) {
    return @{
      managed_pid = 0
      profile = ""
      started_at = ""
      last_external_healthy = ""
    }
  }
  try {
    $raw = Get-Content $StatePath -Raw
    if (-not $raw) {
      return @{
        managed_pid = 0
        profile = ""
        started_at = ""
        last_external_healthy = ""
      }
    }
    $parsed = $raw | ConvertFrom-Json
    return @{
      managed_pid = [int]($parsed.managed_pid | ForEach-Object { $_ })
      profile = [string]($parsed.profile | ForEach-Object { $_ })
      started_at = [string]($parsed.started_at | ForEach-Object { $_ })
      last_external_healthy = [string]($parsed.last_external_healthy | ForEach-Object { $_ })
    }
  }
  catch {
    return @{
      managed_pid = 0
      profile = ""
      started_at = ""
      last_external_healthy = ""
    }
  }
}

function Save-State {
  param([hashtable]$State)
  $json = $State | ConvertTo-Json -Depth 4
  Set-Content -Path $StatePath -Value $json
}

function Get-ManagedProcess {
  param([int]$PidValue)
  if ($PidValue -le 0) {
    return $null
  }
  try {
    return Get-Process -Id $PidValue -ErrorAction Stop
  }
  catch {
    return $null
  }
}

function Stop-ManagedProcess {
  param([int]$PidValue)
  if ($PidValue -le 0) {
    return
  }
  try {
    Stop-Process -Id $PidValue -Force -ErrorAction Stop
    Write-AutopilotLog -Message ("Stopped managed process pid={0}" -f $PidValue)
  }
  catch {
    Write-AutopilotLog -Message ("Failed stopping pid={0}: {1}" -f $PidValue, $_.Exception.Message) -Level "WARN"
  }
}

function Remove-CloudEnv {
  Remove-Item Env:OPENAI_COMPAT_BASE_URL -ErrorAction SilentlyContinue
  Remove-Item Env:OPENAI_COMPAT_API_KEY -ErrorAction SilentlyContinue
  Remove-Item Env:OPENAI_COMPAT_MODEL -ErrorAction SilentlyContinue
}

function Apply-CommonFastEnv {
  $env:REASONING_MODE = "fast"
  $env:STRICT_RESPONSE_MODE = "false"
  $env:SELF_EVALUATION_ENABLED = "false"
  $env:RESPONSE_MATCH_MODEL_ENABLED = "false"
  $env:MODEL_MAX_RETRIES = "0"
  $env:MODEL_TIMEOUT_SECONDS = "30"
  $env:MEMORY_TOP_K = "2"
  $env:MEMORY_CONTEXT_MAX_TOKENS = "120"
  $env:MEMORY_LONG_TERM_EVERY_N_MESSAGES = "5"
  $env:DEFAULT_ANSWER_STYLE = "relational"
  $env:PRIVACY_REDACTION_ENABLED = "true"
  $env:PRIVACY_REMOVE_SENSITIVE_CONTEXT_KEYS = "true"
  $env:RELATIONSHIP_MEMORY_TEXT_ENABLED = "false"
  $env:INTERNET_LOOKUP_TIMEOUT_SECONDS = "3.0"
  $env:INTERNET_LOOKUP_MAX_RESULTS = "3"
  $env:INTERNET_LOOKUP_MAX_CHARS = "420"
}

function Apply-Profile {
  param([string]$ProfileName)

  Apply-CommonFastEnv
  Remove-CloudEnv

  if ($ProfileName -eq "online") {
    $env:MODEL_BACKEND = "local_llama"
    $env:LOCAL_MODEL_PATH = $OnlineModelPath
    $env:LOCAL_MODEL_THREADS = "8"
    $env:LOCAL_MODEL_MAX_TOKENS = "120"
    $env:INTERNET_LOOKUP_ENABLED = "true"
    return
  }

  if ($OfflineMode -eq "local_light") {
    if (-not (Test-Path $LightModelPath)) {
      Write-AutopilotLog -Message ("Offline light model missing, fallback heuristic path used: {0}" -f $LightModelPath) -Level "WARN"
      $env:MODEL_BACKEND = "heuristic"
      $env:INTERNET_LOOKUP_ENABLED = "false"
      return
    }
    $env:MODEL_BACKEND = "local_llama"
    $env:LOCAL_MODEL_PATH = $LightModelPath
    $env:LOCAL_MODEL_THREADS = "6"
    $env:LOCAL_MODEL_MAX_TOKENS = "80"
    $env:INTERNET_LOOKUP_ENABLED = "true"
    return
  }

  $env:MODEL_BACKEND = "heuristic"
  $env:INTERNET_LOOKUP_ENABLED = "false"
}

function Get-DesiredProfile {
  $now = (Get-Date).TimeOfDay
  $from = [TimeSpan]::Parse($OnlineFrom)
  $to = [TimeSpan]::Parse($OnlineTo)
  $inWindow = $false

  if ($from -le $to) {
    $inWindow = ($now -ge $from -and $now -le $to)
  }
  else {
    $inWindow = ($now -ge $from -or $now -le $to)
  }

  if ($inWindow) {
    return "online"
  }
  return "offline"
}

function Start-ManagedServer {
  param([string]$ProfileName)

  if ($ProfileName -eq "online" -and -not (Test-Path $OnlineModelPath)) {
    throw ("Online model not found: {0}" -f $OnlineModelPath)
  }

  Apply-Profile -ProfileName $ProfileName

  if ($env:MODEL_BACKEND -eq "local_llama") {
    & ".\.venv\Scripts\python" -c "import llama_cpp" 2>$null
    if ($LASTEXITCODE -ne 0) {
      throw "llama-cpp-python is not importable in .venv."
    }
  }

  $pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
  if (-not (Test-Path $pythonExe)) {
    throw ("Python executable not found: {0}" -f $pythonExe)
  }

  $proc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", $BindAddress, "--port", "$Port") `
    -WorkingDirectory $repoRoot `
    -PassThru `
    -WindowStyle Hidden

  Write-AutopilotLog -Message ("Started server pid={0} profile={1} backend={2}" -f $proc.Id, $ProfileName, $env:MODEL_BACKEND)
  return @{
    managed_pid = $proc.Id
    profile = $ProfileName
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    last_external_healthy = ""
  }
}

Write-AutopilotLog -Message ("Autopilot started host={0} port={1} online={2}-{3} offline_mode={4}" -f $BindAddress, $Port, $OnlineFrom, $OnlineTo, $OfflineMode)

$state = Load-State
$externalHealthyLogged = $false

while ($true) {
  $desiredProfile = Get-DesiredProfile
  $healthOk = Test-Health
  $managedProc = Get-ManagedProcess -PidValue ([int]$state.managed_pid)

  if ($null -ne $managedProc) {
    $profileChanged = ([string]$state.profile -ne $desiredProfile)
    $startedAt = $null
    if ([string]$state.started_at) {
      try {
        $startedAt = [datetime]::Parse([string]$state.started_at)
      }
      catch {
        $startedAt = $null
      }
    }
    $warmupExpired = $true
    if ($startedAt -ne $null) {
      $uptimeSec = ((Get-Date).ToUniversalTime() - $startedAt.ToUniversalTime()).TotalSeconds
      $warmupExpired = ($uptimeSec -ge $WarmupSeconds)
    }

    if ($profileChanged) {
      Write-AutopilotLog -Message ("Profile switch needed: {0} -> {1}" -f $state.profile, $desiredProfile)
      Stop-ManagedProcess -PidValue ([int]$state.managed_pid)
      $state = Start-ManagedServer -ProfileName $desiredProfile
      Save-State -State $state
    }
    elseif (-not $healthOk -and $warmupExpired) {
      Write-AutopilotLog -Message ("Health failed for managed pid={0}, restarting" -f $state.managed_pid) -Level "WARN"
      Stop-ManagedProcess -PidValue ([int]$state.managed_pid)
      $state = Start-ManagedServer -ProfileName $desiredProfile
      Save-State -State $state
    }
  }
  else {
    if ($healthOk) {
      if (-not $externalHealthyLogged) {
        Write-AutopilotLog -Message "Healthy external server detected; autopilot will monitor only."
        $externalHealthyLogged = $true
      }
      $state.last_external_healthy = (Get-Date).ToUniversalTime().ToString("o")
      Save-State -State $state
    }
    else {
      $externalHealthyLogged = $false
      $state = Start-ManagedServer -ProfileName $desiredProfile
      Save-State -State $state
    }
  }

  if ($RunOnce) {
    break
  }
  Start-Sleep -Seconds $PollSeconds
}

Write-AutopilotLog -Message "Autopilot run completed."
