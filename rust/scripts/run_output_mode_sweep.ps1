# Compare spatial / GoO movement-output linking modes (50k steps each).
# Usage (from repo root):  pwsh rust/scripts/run_output_mode_sweep.ps1
# Optional:               pwsh rust/scripts/run_output_mode_sweep.ps1 -Steps 100000

param(
    [int]$Steps = 50000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $repoRoot "rust/Cargo.toml"))) {
    $repoRoot = Split-Path $PSScriptRoot -Parent
}
Set-Location (Join-Path $repoRoot "rust")

function Merge-Config {
    param([string]$BasePath, [string]$OverlayPath, [int]$StepsOverride)
    $base = Get-Content $BasePath -Raw | ConvertFrom-Json
    $overlay = Get-Content $OverlayPath -Raw | ConvertFrom-Json
    foreach ($prop in $overlay.PSObject.Properties) {
        $base | Add-Member -NotePropertyName $prop.Name -NotePropertyValue $prop.Value -Force
    }
    if ($StepsOverride -gt 0) { $base.steps = $StepsOverride }
    $tmp = Join-Path $env:TEMP ("evo_output_mode_" + [guid]::NewGuid().ToString() + ".json")
    $base | ConvertTo-Json -Depth 20 | Set-Content $tmp -Encoding utf8
    return $tmp
}

$spatialRuns = @(
    "spatial_separate_readout",
    "spatial_linear_readout",
    "spatial_hidden_sum",
    "spatial_hex_compass",
    "spatial_legacy_linear"
)
$gooRuns = @(
    "goo_hidden_sum",
    "goo_split_channels"
)

Write-Host "Building evo_movement_spatial and evo_movement_goo..."
cargo build --release --bin evo_movement_spatial --bin evo_movement_goo

foreach ($name in $spatialRuns) {
    $cfg = Merge-Config `
        "configs/output_mode_sweep/_base_spatial.json" `
        "configs/output_mode_sweep/$name.json" `
        $Steps
    Write-Host "`n=== spatial: $name ==="
    & "./target/release/evo_movement_spatial.exe" "--config-file=$cfg"
    Remove-Item $cfg -ErrorAction SilentlyContinue
}

foreach ($name in $gooRuns) {
    $cfg = Merge-Config `
        "configs/output_mode_sweep/_base_goo.json" `
        "configs/output_mode_sweep/$name.json" `
        $Steps
    Write-Host "`n=== goo: $name ==="
    & "./target/release/evo_movement_goo.exe" "--config-file=$cfg"
    Remove-Item $cfg -ErrorAction SilentlyContinue
}

Write-Host "`nDone. Compare final populations under output/output_mode_sweep/"
