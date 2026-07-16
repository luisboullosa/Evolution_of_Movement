# Run all 12 ecology headless configs for ~8 hours wall-clock (sequential).
# Calibrates step rate on goo_fast, then divides the budget across configs.
param(
    [double]$Hours = 8,
    [int]$CalibrationSteps = 1500,
    [double]$SafetyFraction = 0.92
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
$OutRoot = Join-Path $RepoRoot "output\ecology_8hr"
$ReleaseDir = Join-Path $RustDir "target\release"

Set-Location $RustDir
$env:CARGO_TARGET_DIR = "target"

. (Join-Path $PSScriptRoot "headless_sim_progress.ps1")

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogFile = Join-Path $OutRoot "run.log"
$ManifestPath = Join-Path $OutRoot "run_manifest.json"
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

    $calibExe = Join-Path $ReleaseDir "evo_movement_goo.exe"
    $calibCfg = "configs/ecology/goo_fast.json"
    Write-Host ""
    Write-Host "Calibrating on goo_fast ($CalibrationSteps steps)..." -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $calibExe --config-file=$calibCfg --steps=$CalibrationSteps --no-progress --no-write-outputs 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Calibration run failed" }
    $sw.Stop()
    $msPerStep = $sw.Elapsed.TotalMilliseconds / [math]::Max($CalibrationSteps, 1)
    # Conservative: calibration is early-pop; full runs spend most time at high occupancy.
    $msPerStep *= 1.12

    $totalBudgetSec = $Hours * 3600.0 * $SafetyFraction
    $runs = @(
        @{ Name = "perceptron_fast";    Bin = "evo_movement_perceptron"; Config = "configs/ecology/perceptron_fast.json" },
        @{ Name = "perceptron_medium";  Bin = "evo_movement_perceptron"; Config = "configs/ecology/perceptron_medium.json" },
        @{ Name = "perceptron_slow";    Bin = "evo_movement_perceptron"; Config = "configs/ecology/perceptron_slow.json" },
        @{ Name = "mlp_fast";           Bin = "evo_movement_mlp";        Config = "configs/ecology/mlp_fast.json" },
        @{ Name = "mlp_medium";         Bin = "evo_movement_mlp";        Config = "configs/ecology/mlp_medium.json" },
        @{ Name = "mlp_slow";           Bin = "evo_movement_mlp";        Config = "configs/ecology/mlp_slow.json" },
        @{ Name = "spatial_graph_fast"; Bin = "evo_movement_spatial";    Config = "configs/ecology/spatial_graph_fast.json" },
        @{ Name = "spatial_graph_medium"; Bin = "evo_movement_spatial";  Config = "configs/ecology/spatial_graph_medium.json" },
        @{ Name = "spatial_graph_slow"; Bin = "evo_movement_spatial";    Config = "configs/ecology/spatial_graph_slow.json" },
        @{ Name = "goo_fast";           Bin = "evo_movement_goo";        Config = "configs/ecology/goo_fast.json" },
        @{ Name = "goo_medium";         Bin = "evo_movement_goo";        Config = "configs/ecology/goo_medium.json" },
        @{ Name = "goo_slow";           Bin = "evo_movement_goo";        Config = "configs/ecology/goo_slow.json" }
    )

    $perRunSec = $totalBudgetSec / $runs.Count
    $stepsPerRun = [int][math]::Floor(($perRunSec * 1000.0) / $msPerStep)
    $stepsPerRun = [math]::Max($stepsPerRun, 5000)
    $reportInterval = [math]::Max(2000, [int][math]::Floor($stepsPerRun / 50.0))

    $manifest = [ordered]@{
        started_at = $StartedAt.ToString("o")
        hours_budget = $Hours
        safety_fraction = $SafetyFraction
        calibration_steps = $CalibrationSteps
        ms_per_step_estimate = [math]::Round($msPerStep, 3)
        steps_per_run = $stepsPerRun
        report_interval = $reportInterval
        total_sim_steps_planned = $stepsPerRun * $runs.Count
        per_run_budget_sec = [math]::Round($perRunSec, 1)
        output_root = $OutRoot
        runs = @()
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding UTF8

    Write-Host ""
    Write-Host ("Budget: {0:N1}h wall × {1:P0} safety => {2:N0} steps/run × {3} configs = {4:N0} total sim steps" -f `
        $Hours, $SafetyFraction, $stepsPerRun, $runs.Count, ($stepsPerRun * $runs.Count)) -ForegroundColor Green
    Write-Host ("Estimate: {0:N1} ms/step, ~{1:N0}s per run (~{2:hh\:mm\:ss} each)" -f `
        $msPerStep, $perRunSec, [TimeSpan]::FromSeconds($perRunSec))

    foreach ($run in $runs) {
        $exe = Join-Path $ReleaseDir "$($run.Bin).exe"
        if (-not (Test-Path $exe)) {
            throw "Missing binary: $exe"
        }
        $outDir = Join-Path $OutRoot $run.Name
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null

        Write-Host ""
        Write-Host "========== $($run.Name) ($stepsPerRun steps) ==========" -ForegroundColor Cyan
        $runStarted = Get-Date
        $code = Invoke-HeadlessSimWithProgress -Exe $exe -Config $run.Config -Activity "$($run.Name) ($stepsPerRun steps)" `
            -ExtraArgs @(
                "--steps=$stepsPerRun",
                "--report-interval=$reportInterval",
                "--output-run-dir=$outDir"
            )
        if ($code -ne 0) {
            throw "Simulation failed: $($run.Name)"
        }
        $elapsed = (Get-Date) - $runStarted
        $snap = Join-Path $outDir "snapshot.cbor"
        $snapMb = if (Test-Path $snap) { [math]::Round((Get-Item $snap).Length / 1MB, 2) } else { 0 }
        Write-Host ("Finished {0} in {1} (snapshot {2} MB)" -f $run.Name, $elapsed.ToString("hh\:mm\:ss"), $snapMb) -ForegroundColor Green

        $manifest.runs += [ordered]@{
            name = $run.Name
            steps = $stepsPerRun
            duration_sec = [math]::Round($elapsed.TotalSeconds, 1)
            snapshot_mb = $snapMb
        }
    }

    $totalElapsed = (Get-Date) - $StartedAt
    $manifest.finished_at = (Get-Date).ToString("o")
    $manifest.total_wall_sec = [math]::Round($totalElapsed.TotalSeconds, 1)
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $ManifestPath -Encoding UTF8

    Write-Host ""
    Write-Host ("All ecology runs complete in {0}" -f $totalElapsed.ToString("hh\:mm\:ss")) -ForegroundColor Green
    Write-Host "Manifest: $ManifestPath"
} finally {
    Stop-Transcript | Out-Null
}
