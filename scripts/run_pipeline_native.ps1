# Native-minimal founders (per architecture), basal_metabolic_cost=0.05, 100K steps.
$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
$OutDir = Join-Path $RepoRoot "output\pipeline_native"
$Interactive = Join-Path $RustDir "target\release\evo_movement_interactive.exe"

Set-Location $RustDir

. (Join-Path $PSScriptRoot "headless_sim_progress.ps1")

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogFile = Join-Path $OutDir "run.log"
Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "Building release binaries..."
    cargo build --release --features default `
        --bin evo_movement_perceptron `
        --bin evo_movement_mlp `
        --bin evo_movement_spatial `
        --bin evo_movement_goo `
        --bin evo_movement_interactive
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $runs = @(
        @{ Bin = "evo_movement_perceptron"; Config = "configs/pipeline_native/perceptron.json" },
        @{ Bin = "evo_movement_mlp";       Config = "configs/pipeline_native/mlp.json" },
        @{ Bin = "evo_movement_spatial";    Config = "configs/pipeline_native/spatial_graph.json" },
        @{ Bin = "evo_movement_goo";        Config = "configs/pipeline_native/goo.json" }
    )

    $ReleaseDir = Join-Path $RustDir "target\release"

    foreach ($run in $runs) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($run.Config)
        $exe = Join-Path $ReleaseDir "$($run.Bin).exe"
        if (-not (Test-Path $exe)) {
            Write-Error "Missing binary: $exe (run cargo build --release first)"
            exit 1
        }
        Write-Host ""
        Write-Host "========== $name ($($run.Bin)) ==========" -ForegroundColor Cyan
        $started = Get-Date
        $code = Invoke-HeadlessSimWithProgress -Exe $exe -Config $run.Config -Activity "$name ($($run.Bin))"
        if ($code -ne 0) {
            Write-Error "Simulation failed: $($run.Bin)"
            exit $LASTEXITCODE
        }
        $snap = Join-Path $OutDir $name "snapshot.cbor"
        if (-not (Test-Path $snap)) {
            Write-Error "Expected snapshot missing: $snap"
            exit 1
        }
        $elapsed = (Get-Date) - $started
        Write-Host "Finished $name in $($elapsed.ToString('hh\:mm\:ss')) (snapshot $([math]::Round((Get-Item $snap).Length/1MB,2)) MB)" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "All simulations complete. Launching interactive arena..." -ForegroundColor Cyan
    if (-not (Test-Path $Interactive)) {
        Write-Error "Interactive binary not found at $Interactive"
        exit 1
    }
    & $Interactive --arena=../output/pipeline_native --r=30
} finally {
    Stop-Transcript | Out-Null
}
