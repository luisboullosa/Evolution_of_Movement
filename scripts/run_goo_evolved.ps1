# Long GoO evolution run: 200K steps, 10K pop, higher architecture_mutation_rate.
$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
$OutDir = Join-Path $RepoRoot "output\pipeline_goo_evolved"
$Exe = Join-Path $RustDir "target\release\evo_movement_goo.exe"
$Config = "configs/pipeline_goo_evolved/goo.json"

Set-Location $RustDir
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "goo") | Out-Null

Write-Host "Building evo_movement_goo (release, default features)..."
cargo build --release --features default --bin evo_movement_goo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$run = {
    param($cargoArgs)
    if (Test-Path $Exe) {
        & $Exe @cargoArgs
    } else {
        cargo run --release --features default --bin evo_movement_goo -- @cargoArgs
    }
}

Write-Host ""
Write-Host "GoO evolved run: 200K steps, arch_mutation_rate=0.03, max_brain_nodes=48" -ForegroundColor Cyan
Write-Host "Output: $OutDir\goo" -ForegroundColor Cyan
$started = Get-Date
& $run @("--config-file=$Config")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$snap = Join-Path $OutDir "goo\snapshot.cbor"
if (-not (Test-Path $snap)) {
    Write-Error "Snapshot missing: $snap"
    exit 1
}

$elapsed = (Get-Date) - $started
Write-Host ""
Write-Host "Done in $($elapsed.ToString('hh\:mm\:ss')). Snapshot $([math]::Round((Get-Item $snap).Length/1MB,2)) MB" -ForegroundColor Green
Write-Host ""
Write-Host "View in arena:" -ForegroundColor Cyan
Write-Host "  cd rust"
Write-Host "  .\target\release\evo_movement_interactive.exe --load-snapshot=../output/pipeline_goo_evolved/goo/snapshot.cbor"
