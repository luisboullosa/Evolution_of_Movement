# Plateau-80 medium speed, max_pop 2000 — all 4 architectures, ~8h wall-clock.
param(
    [double]$Hours = 8,
    [int]$CalibrationSteps = 2000,
    [double]$SafetyFraction = 0.92
)

$ErrorActionPreference = "Stop"
$RustDir = (Resolve-Path (Join-Path $PSScriptRoot "..\rust")).Path
$RepoRoot = Split-Path $RustDir -Parent
$OutRoot = Join-Path $RepoRoot "output\ecology_plateau80_medium_2k_8hr"
$ReleaseDir = Join-Path $RustDir "target\release"
$ConfigDir = "configs/ecology_plateau80/medium_2k"

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
    $calibCfg = "$ConfigDir/goo.json"
    Write-Host ""
    Write-Host "Calibrating on plateau80 medium_2k goo ($CalibrationSteps steps)..." -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $calibExe --config-file=$calibCfg --steps=$CalibrationSteps --no-progress --no-write-outputs 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Calibration run failed" }
    $sw.Stop()
    $msPerStep = $sw.Elapsed.TotalMilliseconds / [math]::Max($CalibrationSteps, 1)
    $msPerStep *= 1.10

    $totalBudgetSec = $Hours * 3600.0 * $SafetyFraction
    $runs = @(
        @{ Name = "perceptron";    Bin = "evo_movement_perceptron"; Config = "$ConfigDir/perceptron.json" },
        @{ Name = "mlp";           Bin = "evo_movement_mlp";        Config = "$ConfigDir/mlp.json" },
        @{ Name = "spatial_graph"; Bin = "evo_movement_spatial";    Config = "$ConfigDir/spatial_graph.json" },
        @{ Name = "goo";           Bin = "evo_movement_goo";        Config = "$ConfigDir/goo.json" }
    )

    $perRunSec = $totalBudgetSec / $runs.Count
    $stepsPerRun = [int][math]::Floor(($perRunSec * 1000.0) / $msPerStep)
    $stepsPerRun = [math]::Max($stepsPerRun, 50000)
    $reportInterval = [math]::Max(5000, [int][math]::Floor($stepsPerRun / 50.0))

    $manifest = [ordered]@{
        profile = "plateau80_medium_2k"
        movement_speed = "medium"
        max_population = 2000
        initial_population = 300
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
    Write-Host ("Plateau-80 medium @ 2k cap | {0:N1}h x {1:P0} => {2:N0} steps/run x 4 arch" -f `
        $Hours, $SafetyFraction, $stepsPerRun) -ForegroundColor Green
    Write-Host ("Estimate: {0:N1} ms/step, ~{1:hh\:mm\:ss} per architecture" -f `
        $msPerStep, [TimeSpan]::FromSeconds($perRunSec))

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
    Write-Host ("All runs complete in {0}" -f $totalElapsed.ToString("hh\:mm\:ss")) -ForegroundColor Green
    Write-Host "Manifest: $ManifestPath"
} finally {
    Stop-Transcript | Out-Null
}
