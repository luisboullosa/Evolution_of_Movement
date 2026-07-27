# DEPRECATED — 1M/10rep far exceeds practical wall clock. Use run_fitness_landscape_2d.ps1 (~48h).
# Full fitness landscape: all architectures, 1M steps, 5 replicates per cell.
param(
    [int]$Steps = 500000,
    [int]$Replicates = 5,
    [double]$CapMultiplier = 1.8,
    [string]$FounderModes = "neutral,random_uniform",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $RepoRoot "output\fitness_landscape_1m_10rep"
}

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$LogFile = Join-Path $OutDir "run.log"
$StartedAt = Get-Date

Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "Building evo_fitness_landscape (release)..." -ForegroundColor Cyan
    cargo build --release --bin evo_fitness_landscape
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Landscape sweep: steps=$Steps replicates=$Replicates founders=$FounderModes" -ForegroundColor Cyan
    Write-Host "72 cells x $($FounderModes.Split(',').Count) founder modes x $Replicates replicates" -ForegroundColor Yellow
    Write-Host "Output: $OutDir" -ForegroundColor Yellow

    .\target\release\evo_fitness_landscape.exe `
        --steps=$Steps `
        --replicates=$Replicates `
        --cap-multiplier=$CapMultiplier `
        --founder-modes="$FounderModes" `
        --output-dir="$OutDir" `
        --base-config-root="configs/ecology_plateau80/medium_100k"

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $elapsed = (Get-Date) - $StartedAt
    Write-Host ""
    Write-Host "Done in $([math]::Round($elapsed.TotalHours, 2)) hours" -ForegroundColor Green
    Write-Host "Open: $OutDir\fitness_landscape_heatmap.html" -ForegroundColor Yellow
}
finally {
    Stop-Transcript | Out-Null
}
