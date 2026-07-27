# Plateau-80 spatial macromolecule blob — 100k steps at fast, medium, and slow locomotion.
$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
$OutRoot = Join-Path $RepoRoot "output\ecology_plateau80_spatial_100k"
$ReleaseDir = Join-Path $RustDir "target\release"
$ConfigDir = "configs/ecology_plateau80/spatial_100k"

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogFile = Join-Path $OutRoot "run.log"
$StartedAt = Get-Date

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "Building release spatial headless binary..." -ForegroundColor Cyan
    cargo build --release --bin evo_movement_spatial
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $exe = Join-Path $ReleaseDir "evo_movement_spatial.exe"
    $runs = @(
        @{ Name = "spatial_fast";   Cfg = "$ConfigDir/spatial_fast.json" },
        @{ Name = "spatial_medium"; Cfg = "$ConfigDir/spatial_medium.json" },
        @{ Name = "spatial_slow";   Cfg = "$ConfigDir/spatial_slow.json" }
    )

    foreach ($run in $runs) {
        Write-Host ""
        Write-Host "=== $($run.Name) (100k steps, plateau80, hex_edge blob) ===" -ForegroundColor Cyan
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $exe --config-file=$($run.Cfg) --no-write-outputs
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $sw.Stop()
        Write-Host "Finished $($run.Name) in $([math]::Round($sw.Elapsed.TotalMinutes, 1)) min" -ForegroundColor Green
    }

    $elapsed = (Get-Date) - $StartedAt
    Write-Host ""
    Write-Host "All spatial speed runs complete in $([math]::Round($elapsed.TotalMinutes, 1)) min" -ForegroundColor Green
    Write-Host "Outputs: output/ecology_plateau80_spatial_100k/" -ForegroundColor Yellow
}
finally {
    Stop-Transcript | Out-Null
}
