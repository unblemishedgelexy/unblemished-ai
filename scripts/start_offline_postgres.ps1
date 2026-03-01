param(
  [string]$PostgresHost = "127.0.0.1",
  [int]$PostgresPort = 5432,
  [string]$PostgresUser = "postgres",
  [string]$PostgresPassword = "postgres",
  [string]$PostgresDatabase = "humoniod_ai",
  [ValidateSet("heuristic", "local_llama")] [string]$ModelBackend = "heuristic",
  [string]$LocalModelPath = "",
  [switch]$NoReload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python"
if (-not (Test-Path $python)) {
  throw "Python virtual environment not found at .venv. Create venv and install requirements first."
}

$escapedPassword = [uri]::EscapeDataString($PostgresPassword)
$env:DATABASE_DRIVER = "postgres"
$env:POSTGRES_DSN = "postgresql://$PostgresUser`:$escapedPassword@$PostgresHost`:$PostgresPort/$PostgresDatabase?sslmode=disable"

$env:MODEL_BACKEND = $ModelBackend
$env:REASONING_MODE = "fast"
$env:MODEL_TIMEOUT_SECONDS = "30"
$env:MODEL_MAX_RETRIES = "1"
$env:SELF_EVALUATION_ENABLED = "false"
$env:STRICT_RESPONSE_MODE = "false"
$env:INTERNET_LOOKUP_ENABLED = "false"
$env:RESPONSE_MATCH_MODEL_ENABLED = "false"

# Offline run should not depend on external OpenAI-compatible APIs.
Remove-Item Env:OPENAI_COMPAT_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_MODEL -ErrorAction SilentlyContinue

if ($ModelBackend -eq "local_llama") {
  if (-not $LocalModelPath) {
    $LocalModelPath = "D:\kanchana-ai\ai-model\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  }
  if (-not (Test-Path $LocalModelPath)) {
    throw "Local llama model file not found: $LocalModelPath"
  }
  $env:LOCAL_MODEL_PATH = $LocalModelPath
  $env:LOCAL_MODEL_THREADS = "8"
  $env:LOCAL_MODEL_MAX_TOKENS = "120"
  & $python -c "import llama_cpp" 2>$null
  if ($LASTEXITCODE -ne 0) {
    throw "llama-cpp-python is not installed/importable in .venv. Install requirements first."
  }
}

Write-Host ""
Write-Host "Offline Postgres startup configuration"
Write-Host ("  DATABASE_DRIVER={0}" -f $env:DATABASE_DRIVER)
Write-Host ("  MODEL_BACKEND={0}" -f $env:MODEL_BACKEND)
Write-Host ("  POSTGRES_DSN={0}" -f $env:POSTGRES_DSN)
Write-Host ""

# Verify Postgres connectivity before starting uvicorn.
@'
import asyncio
import os
import sys

async def main():
    dsn = os.environ.get("POSTGRES_DSN", "")
    try:
        import asyncpg
        conn = await asyncpg.connect(dsn=dsn, timeout=8)
        value = await conn.fetchval("SELECT 1")
        await conn.close()
        print(f"postgres_connectivity=ok result={value}")
    except Exception as exc:
        print(f"postgres_connectivity=failed error={type(exc).__name__}: {exc}")
        sys.exit(1)

asyncio.run(main())
'@ | & $python -

if ($LASTEXITCODE -ne 0) {
  throw "Postgres connectivity check failed. Fix host/user/password/database and retry."
}

if ($NoReload.IsPresent) {
  & $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
}
else {
  & $python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
}
