# Build brain_playground WASM for docs/brain_graph_playground.html and fitness landscape probe
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$wasmCrate = Join-Path $root "rust\brain_playground_wasm"
$outDirs = @(
    (Join-Path $root "docs\brain_playground_wasm"),
    (Join-Path $root "rust\assets\brain_playground_wasm")
)

if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Error "wasm-pack not found. Install with: cargo install wasm-pack"
}

Push-Location $wasmCrate
try {
    foreach ($outDir in $outDirs) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        wasm-pack build --target web --release --out-dir $outDir
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "WASM bundle written to $outDir"
    }
} finally {
    Pop-Location
}
