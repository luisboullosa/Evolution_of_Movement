# Staged evolution hunt: each replicate runs 10k -> 50k -> 100k -> 200k from snapshots.
# All four architectures; all cells at every stage; promotion is reported but not used to filter.
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
    $OutDir = Join-Path $RepoRoot "output\evolution_hunt_staged_all"
}

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogFile = Join-Path $OutDir "run.log"
$StartedAt = Get-Date

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "=== Staged evolution hunt (all architectures) ===" -ForegroundColor Cyan
    Write-Host "Per replicate: 10K (fresh) -> 50K -> 100K -> 200K (continues from snapshots)" -ForegroundColor Cyan
    Write-Host "Stages: $SweepStages" -ForegroundColor Yellow
    Write-Host "Perceptron + MLP + Spatial (hex+cardinal) + GoO · profile=evolution_hunt" -ForegroundColor Cyan
    Write-Host "Replicates=$Replicates founders=$FounderModes" -ForegroundColor Yellow
    Write-Host "Output: $OutDir" -ForegroundColor Yellow
    Write-Host "  stage_10000/ snapshots/ per rep+seed -> stage_50000/ continues from them" -ForegroundColor DarkGray
    Write-Host "  combined heatmap at root (latest stage per cell)" -ForegroundColor DarkGray
    if ($Resume) { Write-Host "Mode: RESUME" -ForegroundColor Magenta }

    Write-Host "Building evo_fitness_landscape (release)..." -ForegroundColor Cyan
    cargo build --release --bin evo_fitness_landscape --features sqlite
    $built = ($LASTEXITCODE -eq 0)

    $resumeArg = if ($Resume) { @("--resume") } else { @() }
    $stagedArgs = @(
        "--staged-sweep",
        "--sweep-stages=$SweepStages",
        "--replicates=$Replicates",
        "--cap-multiplier=$CapMultiplier",
        "--founder-modes=$FounderModes",
        "--sweep-output-modes",
        "--sweep-profile=evolution_hunt",
        "--output-dir=$OutDir",
        "--base-config-root=configs/ecology_plateau80/evolution_hunt"
    ) + $resumeArg

    if ($built) {
        $exe = Join-Path $RustDir "target\release\evo_fitness_landscape.exe"
        & $exe @stagedArgs
    } else {
        Write-Host "Build failed (exe may be locked); using cargo run..." -ForegroundColor Yellow
        cargo run --release --bin evo_fitness_landscape --features sqlite -- @stagedArgs
    }

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $elapsed = (Get-Date) - $StartedAt
    Write-Host ""
    Write-Host "Done in $([math]::Round($elapsed.TotalHours, 2)) hours" -ForegroundColor Green
    Write-Host "Combined: $OutDir\fitness_landscape_heatmap.html" -ForegroundColor Yellow
    Write-Host "Stage reports: $OutDir\stage_10000\promotion_report.txt (etc.)" -ForegroundColor Yellow
    Write-Host "Probe timelines (optional, while sweep runs): pwsh scripts/build_probe_timelines.ps1 -OutDir $OutDir" -ForegroundColor DarkGray
    Write-Host "Viewer + rebuild button: python scripts/serve_fitness_landscape.py $OutDir" -ForegroundColor DarkGray
}
finally {
    Stop-Transcript | Out-Null
}
