Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

& ".\.venv\Scripts\python" -m pip install "huggingface_hub>=0.24.0"
& ".\.venv\Scripts\hf.exe" download `
  bartowski/Llama-3.2-3B-Instruct-GGUF `
  Llama-3.2-3B-Instruct-Q4_K_M.gguf `
  --local-dir "D:\kanchana-ai\ai-model\models"
