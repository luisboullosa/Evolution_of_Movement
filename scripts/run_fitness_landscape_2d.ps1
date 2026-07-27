# Fitness landscape sized for ~48h wall clock (2 days).
# Calibrated from v2: 432 runs x 30k steps ~= 14h release build.
#   432 runs x 100k steps ~= 47h  (3 replicates x 2 founder modes x 72 cells)
# Alternative 2-day profile: 30k steps x 10 replicates (1440 runs, similar total compute).
param(
    [int]$Steps = 100000,
    [int]$Replicates = 3,
    [double]$CapMultiplier = 1.8,
    [string]$FounderModes = "neutral,random_uniform",
    [string]$OutDir = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $RepoRoot "output\fitness_landscape_2d_100k_3rep"
}

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogFile = Join-Path $OutDir "run.log"
$StartedAt = Get-Date

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    $founderCount = ($FounderModes -split ',').Count
    $totalRuns = 72 * $founderCount * $Replicates

    Write-Host "=== 2-day fitness landscape sweep ===" -ForegroundColor Cyan
    Write-Host "Budget target: ~48h wall clock" -ForegroundColor Yellow
    Write-Host "Steps=$Steps replicates=$Replicates founders=$FounderModes" -ForegroundColor Yellow
    Write-Host "72 cells x $founderCount founder modes x $Replicates replicates = $totalRuns runs" -ForegroundColor Yellow
    Write-Host "Output: $OutDir" -ForegroundColor Yellow
    if ($Resume) { Write-Host "Mode: RESUME from checkpoint" -ForegroundColor Magenta }

    Write-Host "Building evo_fitness_landscape (release)..." -ForegroundColor Cyan
    cargo build --release --bin evo_fitness_landscape
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $resumeArg = if ($Resume) { @("--resume") } else { @() }
    .\target\release\evo_fitness_landscape.exe `
        --steps=$Steps `
        --replicates=$Replicates `
        --cap-multiplier=$CapMultiplier `
        --founder-modes="$FounderModes" `
        --output-dir="$OutDir" `
        --base-config-root="configs/ecology_plateau80/medium_100k" `
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
