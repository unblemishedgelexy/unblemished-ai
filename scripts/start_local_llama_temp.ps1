Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$env:MODEL_BACKEND = "local_llama"
$env:REASONING_MODE = "fast"
$env:LOCAL_MODEL_PATH = "D:\kanchana-ai\ai-model\models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$env:LOCAL_MODEL_THREADS = "8"
$env:LOCAL_MODEL_MAX_TOKENS = "80"
$env:MEMORY_CONTEXT_MAX_TOKENS = "120"
$env:STRICT_RESPONSE_MODE = "false"
$env:DEFAULT_ANSWER_STYLE = "flirty"
$env:RESPONSE_MATCH_MODEL_ENABLED = "false"
$env:MODEL_TIMEOUT_SECONDS = "30"
$env:MODEL_MAX_RETRIES = "0"
$env:SELF_EVALUATION_ENABLED = "false"
$env:RESPONSE_MATCH_MODEL_TIMEOUT_SECONDS = "8.0"
$env:RESPONSE_MATCH_MODEL_MAX_RETRIES = "0"
$env:INTERNET_LOOKUP_ENABLED = "true"
$env:INTERNET_LOOKUP_TIMEOUT_SECONDS = "3.0"
$env:INTERNET_LOOKUP_MAX_RESULTS = "3"
$env:INTERNET_LOOKUP_MAX_CHARS = "420"
$env:PRIVACY_REDACTION_ENABLED = "true"
$env:PRIVACY_REMOVE_SENSITIVE_CONTEXT_KEYS = "true"
$env:RELATIONSHIP_MEMORY_TEXT_ENABLED = "false"

if (-not (Test-Path $env:LOCAL_MODEL_PATH)) {
  throw "Model file not found: $($env:LOCAL_MODEL_PATH)"
}

& ".\.venv\Scripts\python" -c "import llama_cpp" 2>$null
if ($LASTEXITCODE -ne 0) {
  throw "llama-cpp-python is not installed or not importable. Install toolchain / package first."
}

Remove-Item Env:OPENAI_COMPAT_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:OPENAI_COMPAT_MODEL -ErrorAction SilentlyContinue

& ".\.venv\Scripts\python" -m uvicorn app.main:app --reload
