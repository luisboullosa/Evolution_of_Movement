# Build probe timeline JSON from staged sweep snapshots (read-only; safe while sweep runs).
param(
    [string]$OutDir = "",
    [string]$Stages = "0,10000,50000",
    [string]$BaseConfigRoot = "configs/ecology_plateau80/evolution_hunt",
    [switch]$SweepOutputModes,
    [double]$CapMultiplier = 1.8
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $RepoRoot "output\evolution_hunt_staged_all"
}

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

Write-Host "=== Build probe timelines ===" -ForegroundColor Cyan
Write-Host "Output: $OutDir" -ForegroundColor Yellow
Write-Host "Stages: $Stages" -ForegroundColor Yellow
Write-Host "Does not rebuild evo_fitness_landscape — safe alongside a running sweep." -ForegroundColor DarkGray

Write-Host "Building evo_probe_timeline (release)..." -ForegroundColor Cyan
cargo build --release --bin evo_probe_timeline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$exe = Join-Path $RustDir "target\release\evo_probe_timeline.exe"
$args = @(
    "--output-dir=$OutDir",
    "--stages=$Stages",
    "--profile=evolution_hunt",
    "--base-config-root=$BaseConfigRoot",
    "--cap-multiplier=$CapMultiplier",
    "--from-state"
)
if ($SweepOutputModes) { $args += "--sweep-output-modes" }

& $exe @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Timelines: $OutDir\probe_timelines\" -ForegroundColor Green
Write-Host "Manifest:  $OutDir\probe_timelines\manifest.json" -ForegroundColor Green
Write-Host "Open fitness_landscape_heatmap.html and select a cell to view timeline sweeps." -ForegroundColor DarkGray
Write-Host "Serve with rebuild API: python scripts/serve_fitness_landscape.py `"$OutDir`"" -ForegroundColor DarkGray
