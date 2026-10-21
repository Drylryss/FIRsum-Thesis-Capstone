$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Run .\setup.ps1 first.'
}
Write-Host 'FIRmSUM: open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.'
& '.\.venv\Scripts\python.exe' inference.py
