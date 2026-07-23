# Spatial + GoO fitness landscape with movement_output_mode sweep.
# Same complexity grid as run_fitness_landscape_2d.ps1, but:
#   - only spatial and goo architectures
#   - each cell replicated across output-linking modes
#     spatial: separate_readout, linear_readout, hidden_sum, hex_compass
#     goo: hidden_sum, split_channels
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
    $OutDir = Join-Path $RepoRoot "output\fitness_landscape_2d_100k_3rep_spatial_goo_modes"
}

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogFile = Join-Path $OutDir "run.log"
$StartedAt = Get-Date

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    $founderCount = ($FounderModes -split ',').Count
    # 24 spatial + 20 goo cells, x4 + x2 output modes = 136 cells
    $totalRuns = 136 * $founderCount * $Replicates

    Write-Host "=== Spatial + GoO output-mode fitness landscape ===" -ForegroundColor Cyan
    Write-Host "Steps=$Steps replicates=$Replicates founders=$FounderModes" -ForegroundColor Yellow
    Write-Host "136 cells x $founderCount founder modes x $Replicates replicates = $totalRuns runs" -ForegroundColor Yellow
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
        --architectures=spatial,goo `
        --sweep-output-modes `
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
