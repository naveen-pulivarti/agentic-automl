# Run the full benchmark set: deterministic baseline first, then the reasoning
# layer, sequentially so the two never contend for the GPU or the CPU.
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
$py = Join-Path $proj ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

Write-Output "=== 1/2  deterministic baseline (LLM endpoint unreachable) ==="
$env:OLLAMA_BASE_URL = "http://127.0.0.1:1"
Remove-Item Env:AUTOML_LLM_PROVIDER -ErrorAction SilentlyContinue
& $py (Join-Path $proj "benchmarks\bench.py") v3_heuristic

Write-Output ""
Write-Output "=== 2/2  reasoning layer (local llama3.1) ==="
Remove-Item Env:OLLAMA_BASE_URL -ErrorAction SilentlyContinue
$env:AUTOML_LLM_PROVIDER = "ollama"
$env:AUTOML_LLM_MODEL = "llama3.1"
& $py (Join-Path $proj "benchmarks\bench.py") v3_llama31

Write-Output ""
Write-Output "=== done ==="
