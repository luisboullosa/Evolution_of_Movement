# Staged GoO + Spatial evolution hunt: 10k -> 50k -> 100k -> 200k with promotion.
# For all four architectures, use run_evolution_hunt_all.ps1 instead.
param(
    [int]$Replicates = 3,
    [double]$CapMultiplier = 1.8,
    [string]$FounderModes = "random_uniform,random_small",
    [string]$SweepStages = "10000,50000,100000,200000",
    [string]$OutDir = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $RepoRoot "output\evolution_hunt_staged_spatial_goo"
}

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogFile = Join-Path $OutDir "run.log"
$StartedAt = Get-Date

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "=== Staged evolution hunt: Spatial + GoO ===" -ForegroundColor Cyan
    Write-Host "Stages: $SweepStages" -ForegroundColor Yellow
    Write-Host "Replicates=$Replicates founders=$FounderModes profile=evolution_hunt" -ForegroundColor Yellow
    Write-Host "Output: $OutDir" -ForegroundColor Yellow
    Write-Host "Live viewer: python -m http.server 8765 in output dir" -ForegroundColor DarkGray
    if ($Resume) { Write-Host "Mode: RESUME" -ForegroundColor Magenta }

    Write-Host "Building evo_fitness_landscape (release)..." -ForegroundColor Cyan
    cargo build --release --bin evo_fitness_landscape --features sqlite
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $resumeArg = if ($Resume) { @("--resume") } else { @() }
    $exe = Join-Path $RustDir "target\release\evo_fitness_landscape.exe"
    & $exe `
        --staged-sweep `
        --sweep-stages="$SweepStages" `
        --replicates=$Replicates `
        --cap-multiplier=$CapMultiplier `
        --founder-modes="$FounderModes" `
        --architectures=spatial,goo `
        --sweep-output-modes `
        --sweep-profile=evolution_hunt `
        --output-dir="$OutDir" `
        --base-config-root="configs/ecology_plateau80/evolution_hunt" `
        @resumeArg

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $elapsed = (Get-Date) - $StartedAt
    Write-Host ""
    Write-Host "Done in $([math]::Round($elapsed.TotalHours, 2)) hours" -ForegroundColor Green
    Write-Host "Open: $OutDir\fitness_landscape_heatmap.html" -ForegroundColor Yellow
}
finally {
    Stop-Transcript | Out-Null
}
