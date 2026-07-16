# Plateau-80 medium speed, 100k steps, structural duplication — all 4 architectures.
$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
$OutRoot = Join-Path $RepoRoot "output\ecology_plateau80_medium_100k"
$ReleaseDir = Join-Path $RustDir "target\release"
$ConfigDir = "configs/ecology_plateau80/medium_100k"

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogFile = Join-Path $OutRoot "run.log"
$StartedAt = Get-Date

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "Building release headless binaries..." -ForegroundColor Cyan
    cargo build --release --features default `
        --bin evo_movement_perceptron `
        --bin evo_movement_mlp `
        --bin evo_movement_spatial `
        --bin evo_movement_goo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $runs = @(
        @{ Name = "perceptron";    Exe = "evo_movement_perceptron.exe"; Cfg = "$ConfigDir/perceptron.json" },
        @{ Name = "mlp";           Exe = "evo_movement_mlp.exe";        Cfg = "$ConfigDir/mlp.json" },
        @{ Name = "spatial_graph"; Exe = "evo_movement_spatial.exe";    Cfg = "$ConfigDir/spatial_graph.json" },
        @{ Name = "goo";           Exe = "evo_movement_goo.exe";       Cfg = "$ConfigDir/goo.json" }
    )

    foreach ($run in $runs) {
        $exe = Join-Path $ReleaseDir $run.Exe
        Write-Host ""
        Write-Host "=== $($run.Name) (100k steps, plateau80 medium, duplication) ===" -ForegroundColor Cyan
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $exe --config-file=$($run.Cfg) --no-write-outputs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $sw.Stop()
        Write-Host "Finished $($run.Name) in $([math]::Round($sw.Elapsed.TotalMinutes, 1)) min" -ForegroundColor Green
    }

    $elapsed = (Get-Date) - $StartedAt
    Write-Host ""
    Write-Host "All runs complete in $([math]::Round($elapsed.TotalMinutes, 1)) min" -ForegroundColor Green
    Write-Host "Arena: output/ecology_plateau80_medium_100k" -ForegroundColor Yellow
}
finally {
    Stop-Transcript | Out-Null
}
