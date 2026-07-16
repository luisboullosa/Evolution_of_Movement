# Build brain_playground WASM for docs/brain_graph_playground.html
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$wasmCrate = Join-Path $root "rust\brain_playground_wasm"
$outDir = Join-Path $root "docs\brain_playground_wasm"

if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Error "wasm-pack not found. Install with: cargo install wasm-pack"
}

Push-Location $wasmCrate
try {
    wasm-pack build --target web --release --out-dir $outDir
} finally {
    Pop-Location
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "WASM bundle written to $outDir"
