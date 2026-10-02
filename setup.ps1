param([switch]$Cpu)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path '.tmp', '.cache' | Out-Null
$env:TEMP = Join-Path $PSScriptRoot '.tmp'
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.cache/python'
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment' }
}
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'Could not install PyTorch' }
uv pip install --python .venv/Scripts/python.exe -r requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Could not install the Python dependencies' }
.venv/Scripts/python.exe cli.py --download
if ($LASTEXITCODE -ne 0) { throw 'Could not download the model' }
