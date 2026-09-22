# typeless-local installer - Windows + AMD GPU (whisper.cpp Vulkan backend)
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "=== typeless-local setup ===" -ForegroundColor Cyan

# --- 1. whisper.cpp official build (Vulkan, runtime bundled) -----------------
$Rel = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/whisper.cpp/releases?per_page=20'
$Nightly = @($Rel | Where-Object { $_.tag_name -match '^b\d+$' })[0]
if (-not $Nightly) { Write-Host "No nightly release found" -ForegroundColor Red; exit 1 }
Write-Host "Latest whisper.cpp build: $($Nightly.tag_name)"

$Asset = @($Nightly.assets | Where-Object { $_.name -eq 'whisper-bin-x64.zip' })[0]
if (-not $Asset) {
  Write-Host "whisper-bin-x64.zip not found; available assets:" -ForegroundColor Yellow
  $Nightly.assets | ForEach-Object { Write-Host "  $($_.name)" }
  exit 1
}

New-Item -ItemType Directory -Force -Path "$Root\bin" | Out-Null
if (-not (Test-Path "$Root\bin\whisper-server.exe")) {
  Write-Host "Downloading $($Asset.name) ..."
  $ZipPath = Join-Path $Root 'bin\whisper-bin-x64.zip'
  Invoke-WebRequest -UseBasicParsing $Asset.browser_download_url -OutFile $ZipPath
  Expand-Archive -Path $ZipPath -DestinationPath "$Root\bin" -Force
  Remove-Item $ZipPath
  # Normalize layout: ensure bin\whisper-server.exe + DLLs exist at top level
  if (-not (Test-Path "$Root\bin\whisper-server.exe")) {
    $Exe = Get-ChildItem "$Root\bin" -Recurse -Filter 'whisper-server.exe' | Select-Object -First 1
    if ($Exe) {
      Copy-Item "$($Exe.DirectoryName)\*" "$Root\bin\" -Force
      Write-Host "Normalized layout from $($Exe.DirectoryName)"
    } else {
      Write-Host "whisper-server.exe not found after extraction" -ForegroundColor Red
      exit 1
    }
  }
}
Write-Host "[ok] whisper.cpp binary ready: $Root\bin\whisper-server.exe" -ForegroundColor Green

# --- 2. Models ---------------------------------------------------------------
# STT models live in the ggerganov/whisper.cpp HF repo; VAD in ggml-org/whisper-vad
$Models = @(
  @{ Name = 'ggml-large-v3-turbo.bin'; Url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin' },
  @{ Name = 'ggml-small.bin';          Url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin' },
  @{ Name = 'ggml-silero-v6.2.0.bin';  Url = 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin' }
)
New-Item -ItemType Directory -Force -Path "$Root\models" | Out-Null
foreach ($M in $Models) {
  $Dest = Join-Path "$Root\models" $M.Name
  if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt 100KB)) {
    Write-Host "[skip] $($M.Name) already present" -ForegroundColor DarkGray
    continue
  }
  Write-Host "Downloading $($M.Name) ..."
  Invoke-WebRequest -UseBasicParsing $M.Url -OutFile $Dest
  Write-Host "[ok] $($M.Name) ($([math]::Round((Get-Item $Dest).Length/1MB,1)) MB)" -ForegroundColor Green
}

# --- 3. Python env (uv) -------------------------------------------------------
Set-Location $Root
$Uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $Uv) {
  Write-Host "uv not found - installing via official installer..." -ForegroundColor Yellow
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
}
uv sync
if ($LASTEXITCODE -ne 0) { Write-Host "uv sync failed" -ForegroundColor Red; exit 1 }
Write-Host "[ok] Python env ready (.venv via uv, pinned in uv.lock)" -ForegroundColor Green

Write-Host ""
Write-Host "=== Done. Next steps ===" -ForegroundColor Cyan
Write-Host "  1. Copy .env.example to .env and fill in your LLM API key"
Write-Host "  2. uv run typeless-server    (start ASR service; auto-starts whisper-server)"
Write-Host "  3. uv run typeless-dictate   (hold F9 to talk, release to type)"
