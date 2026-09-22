# Build whisper.cpp with the Vulkan backend using the MinGW/Strawberry toolchain.
# Produces build/vulkan/bin/whisper-server.exe accelerated on any Vulkan GPU
# (verified on RX 6750 GRE 12GB / gfx1031, which ROCm does not support).
#
# Prereqs: `scoop install vulkan` (LunarG SDK) + gcc/cmake/ninja on PATH.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build-vulkan.ps1
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

# 1. Locate the Vulkan SDK (scoop path by default)
if (-not $env:VULKAN_SDK) {
  $env:VULKAN_SDK = "$env:USERPROFILE\scoop\apps\vulkan\current"
}
if (-not (Test-Path "$env:VULKAN_SDK\bin\glslc.exe")) {
  Write-Host "Vulkan SDK not found. Run: scoop install vulkan" -ForegroundColor Red
  exit 1
}
$env:PATH = "$env:VULKAN_SDK\bin;$env:PATH"
Write-Host "VULKAN_SDK = $env:VULKAN_SDK"

# 2. Source (clone once, keep the MinGW patch applied)
$Src = "$Root\build\whisper-src"
if (-not (Test-Path $Src)) {
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp $Src
}
Push-Location $Src
git checkout -- . 2>$null
git apply "$Root\scripts\mingw-thread-throttling.patch"
if ($LASTEXITCODE -ne 0) { Write-Host "patch failed" -ForegroundColor Red; Pop-Location; exit 1 }
Pop-Location

# 3. Configure + build
cmake -S $Src -B "$Root\build\vulkan" -G Ninja `
  -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release `
  -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON `
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
if ($LASTEXITCODE -ne 0) { exit 1 }
cmake --build "$Root\build\vulkan" --target whisper-server whisper-cli
if ($LASTEXITCODE -ne 0) { exit 1 }

# 4. Bundle MinGW runtime DLLs (executables die with exit 127 without them)
foreach ($dll in @('libstdc++-6.dll','libgcc_s_seh-1.dll','libwinpthread-1.dll','libgomp-1.dll')) {
  $p = "C:\Strawberry\c\bin\$dll"
  if (Test-Path $p) { Copy-Item $p "$Root\build\vulkan\bin\" -Force }
}

# 5. Smoke test (expect: GPU device line + fast encode)
$Sample = "$Root\logs\tts.wav"
if (Test-Path $Sample) {
  Write-Host "`n=== Smoke test (RX 6750 GRE should appear as Vulkan device) ===" -ForegroundColor Cyan
  & "$Root\build\vulkan\bin\whisper-cli.exe" -m "$Root\models\ggml-large-v3-turbo.bin" -t 4 `
    -f $Sample --no-timestamps 2>&1 |
    Select-String -Pattern 'Vulkan devices|encode time|total time'
} else {
  Write-Host "`nNo sample audio found; skipping smoke test." -ForegroundColor Yellow
}
Write-Host "`nDone. WHISPER_BIN=build/vulkan/bin/whisper-server.exe"
