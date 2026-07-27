# Run 12 plateau-80 ecology configs for ~8 hours wall-clock (sequential).
param(
    [double]$Hours = 8,
    [int]$CalibrationSteps = 1500,
    [double]$SafetyFraction = 0.92
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
$OutRoot = Join-Path $RepoRoot "output\ecology_plateau80_8hr"
$ReleaseDir = Join-Path $RustDir "target\release"
$ConfigDir = "configs/ecology_plateau80"

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
    $calibCfg = "$ConfigDir/goo_fast.json"
    Write-Host ""
    Write-Host "Calibrating on plateau80 goo_fast ($CalibrationSteps steps)..." -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $calibExe --config-file=$calibCfg --steps=$CalibrationSteps --no-progress --no-write-outputs 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Calibration run failed" }
    $sw.Stop()
    $msPerStep = $sw.Elapsed.TotalMilliseconds / [math]::Max($CalibrationSteps, 1)
    # Smaller pops than standard ecology; still pad for late-run crowding.
    $msPerStep *= 1.08

    $totalBudgetSec = $Hours * 3600.0 * $SafetyFraction
    $runs = @(
        @{ Name = "perceptron_fast";    Bin = "evo_movement_perceptron"; Config = "$ConfigDir/perceptron_fast.json" },
        @{ Name = "perceptron_medium";  Bin = "evo_movement_perceptron"; Config = "$ConfigDir/perceptron_medium.json" },
        @{ Name = "perceptron_slow";    Bin = "evo_movement_perceptron"; Config = "$ConfigDir/perceptron_slow.json" },
        @{ Name = "mlp_fast";           Bin = "evo_movement_mlp";        Config = "$ConfigDir/mlp_fast.json" },
        @{ Name = "mlp_medium";         Bin = "evo_movement_mlp";        Config = "$ConfigDir/mlp_medium.json" },
        @{ Name = "mlp_slow";           Bin = "evo_movement_mlp";        Config = "$ConfigDir/mlp_slow.json" },
        @{ Name = "spatial_graph_fast"; Bin = "evo_movement_spatial";    Config = "$ConfigDir/spatial_graph_fast.json" },
        @{ Name = "spatial_graph_medium"; Bin = "evo_movement_spatial";  Config = "$ConfigDir/spatial_graph_medium.json" },
        @{ Name = "spatial_graph_slow"; Bin = "evo_movement_spatial";    Config = "$ConfigDir/spatial_graph_slow.json" },
        @{ Name = "goo_fast";           Bin = "evo_movement_goo";        Config = "$ConfigDir/goo_fast.json" },
        @{ Name = "goo_medium";         Bin = "evo_movement_goo";        Config = "$ConfigDir/goo_medium.json" },
        @{ Name = "goo_slow";           Bin = "evo_movement_goo";        Config = "$ConfigDir/goo_slow.json" }
    )

    $perRunSec = $totalBudgetSec / $runs.Count
    $stepsPerRun = [int][math]::Floor(($perRunSec * 1000.0) / $msPerStep)
    $stepsPerRun = [math]::Max($stepsPerRun, 10000)
    $reportInterval = [math]::Max(4000, [int][math]::Floor($stepsPerRun / 50.0))

    $manifest = [ordered]@{
        profile = "plateau80"
        started_at = $StartedAt.ToString("o")
        hours_budget = $Hours
        safety_fraction = $SafetyFraction
        calibration_steps = $CalibrationSteps
        ms_per_step_estimate = [math]::Round($msPerStep, 3)
        steps_per_run = $stepsPerRun
        report_interval = $reportInterval
        total_sim_steps_planned = $stepsPerRun * $runs.Count
        per_run_budget_sec = [math]::Round($perRunSec, 1)
        config_dir = $ConfigDir
        output_root = $OutRoot
        runs = @()
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding UTF8

    Write-Host ""
    Write-Host ("Plateau-80 batch | {0:N1}h wall x {1:P0} safety => {2:N0} steps/run x {3} configs" -f `
        $Hours, $SafetyFraction, $stepsPerRun, $runs.Count) -ForegroundColor Green
    Write-Host ("Estimate: {0:N1} ms/step, ~{1:hh\:mm\:ss} per run" -f $msPerStep, [TimeSpan]::FromSeconds($perRunSec))

    foreach ($run in $runs) {
        $exe = Join-Path $ReleaseDir "$($run.Bin).exe"
        if (-not (Test-Path $exe)) { throw "Missing binary: $exe" }
        $outDir = Join-Path $OutRoot $run.Name
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null

        Write-Host ""
        Write-Host "========== $($run.Name) ($stepsPerRun steps) ==========" -ForegroundColor Cyan
        $runStarted = Get-Date
        $code = Invoke-HeadlessSimWithProgress -Exe $exe -Config $run.Config -Activity "$($run.Name) ($stepsPerRun steps)" `
            -ExtraArgs @("--steps=$stepsPerRun", "--report-interval=$reportInterval", "--output-run-dir=$outDir")
        if ($code -ne 0) { throw "Simulation failed: $($run.Name)" }
        $elapsed = (Get-Date) - $runStarted
        $snap = Join-Path $outDir "snapshot.cbor"
        $snapMb = if (Test-Path $snap) { [math]::Round((Get-Item $snap).Length / 1MB, 2) } else { 0 }
        Write-Host ("Finished {0} in {1} (snapshot {2} MB)" -f $run.Name, $elapsed.ToString("hh\:mm\:ss"), $snapMb) -ForegroundColor Green
        $manifest.runs += [ordered]@{ name = $run.Name; steps = $stepsPerRun; duration_sec = [math]::Round($elapsed.TotalSeconds, 1); snapshot_mb = $snapMb }
    }

    $totalElapsed = (Get-Date) - $StartedAt
    $manifest.finished_at = (Get-Date).ToString("o")
    $manifest.total_wall_sec = [math]::Round($totalElapsed.TotalSeconds, 1)
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $ManifestPath -Encoding UTF8
    Write-Host ""
    Write-Host ("All plateau-80 runs complete in {0}" -f $totalElapsed.ToString("hh\:mm\:ss")) -ForegroundColor Green
} finally {
    Stop-Transcript | Out-Null
}
