param(
    [int]$Steps = 30000,
    [int]$Replicates = 3,
    [double]$CapMultiplier = 1.8,
    [string]$FounderModes = "neutral,random_uniform",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $RepoRoot "output\fitness_landscape_medium"
}

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

Write-Host "Building evo_fitness_landscape..." -ForegroundColor Cyan
cargo build --release --bin evo_fitness_landscape
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running landscape sweep (steps=$Steps, replicates=$Replicates, founders=$FounderModes)..." -ForegroundColor Cyan
.\target\release\evo_fitness_landscape.exe `
  --steps=$Steps `
  --replicates=$Replicates `
  --cap-multiplier=$CapMultiplier `
  --founder-modes="$FounderModes" `
  --output-dir="$OutDir" `
  --base-config-root="configs/ecology_plateau80/medium_100k"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done." -ForegroundColor Green
Write-Host "Open: $OutDir\fitness_landscape_heatmap.html" -ForegroundColor Yellow
