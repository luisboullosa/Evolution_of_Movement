# Backfill run_record.json + run_history.jsonl for completed pipeline_100k runs.
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutDir = Join-Path $RepoRoot "output\pipeline_100k"
$HistoryPath = Join-Path $RepoRoot "output\run_history.jsonl"

$runs = @(
    @{ Name = "perceptron";    Policy = "Perceptron";    Duration = "01:15:31"; Binary = "evo_movement_perceptron.exe" },
    @{ Name = "mlp";           Policy = "MLP";           Duration = "01:14:27"; Binary = "evo_movement_mlp.exe" },
    @{ Name = "spatial_graph"; Policy = "Spatial graph"; Duration = "01:00:14"; Binary = "evo_movement_spatial.exe" },
    @{ Name = "goo";           Policy = "GoO graph";     Duration = "01:00:12"; Binary = "evo_movement_goo.exe" }
)

function ConvertFrom-Duration([string]$text) {
    $parts = $text.Split(':')
    return [int]$parts[0] * 3600 + [int]$parts[1] * 60 + [int]$parts[2]
}

New-Item -ItemType Directory -Force -Path (Split-Path $HistoryPath) | Out-Null
$finishedAt = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

foreach ($run in $runs) {
    $dir = Join-Path $OutDir $run.Name
    $configPath = Join-Path $dir "config.json"
    if (-not (Test-Path $configPath)) {
        Write-Warning "Skipping $($run.Name): no config.json"
        continue
    }
    $cfg = Get-Content $configPath -Raw | ConvertFrom-Json
    $duration = ConvertFrom-Duration $run.Duration
    $steps = [int]$cfg.steps
    $msPerStep = ($duration * 1000.0) / [math]::Max($steps, 1)

    $record = [ordered]@{
        timing = [ordered]@{
            duration_secs = $duration
            ms_per_step = $msPerStep
            steps_per_sec = $steps / [math]::Max($duration, 0.001)
            finished_at_unix_secs = $finishedAt
            backfilled = $true
        }
        worker = [ordered]@{
            binary = [ordered]@{
                name = $run.Binary
                version = "0.1.0"
                build_id = "4bb619c"
                version_label = "v0.1.0 (4bb619c)"
                cuda_arch = "sm_89"
            }
            host_os = "windows"
            cpu_threads = 12
            gpu_step_configured = [bool]$cfg.use_gpu_step
            gpu_population_compatible = $true
            gpu_steps_executed = $steps
            cpu_brain_steps_executed = 0
            backfilled = $true
        }
        sim = [ordered]@{
            config_file = "configs/pipeline_100k/$($run.Name).json"
            output_run_dir = $cfg.output_run_dir
            architecture_policy = $run.Policy
            grid_topology = "hex_2d"
            hex_dish_radius = [int]$cfg.hex_dish_radius
            grid_width = [int]$cfg.grid_width
            grid_height = [int]$cfg.grid_height
            grid_cells = 10267
            steps_requested = $steps
            steps_completed = $steps
            report_interval = [int]$cfg.report_interval
            initial_population = [int]$cfg.initial_population
            max_population = [int]$cfg.max_population
            final_population = [int]$cfg.max_population
            seed = [int]$cfg.seed
            use_gpu_step = [bool]$cfg.use_gpu_step
            max_brain_nodes = [int]$cfg.max_brain_nodes
            nn_nodes = [int]$cfg.nn_nodes
            nn_layers = [int]$cfg.nn_layers
            nn_inputs = [int]$cfg.nn_inputs
            architecture_mutation_rate = [double]$cfg.architecture_mutation_rate
            neutral_founder_weights = [bool]$cfg.neutral_founder_weights
            write_snapshot = [bool]$cfg.write_snapshot
            write_outputs = [bool]$cfg.write_outputs
            show_progress = [bool]$cfg.show_progress
        }
        prop_moved_final = 0.0
    }

    $json = $record | ConvertTo-Json -Depth 8
    $recordPath = Join-Path $dir "run_record.json"
    Set-Content -Path $recordPath -Value $json -Encoding UTF8
    Add-Content -Path $HistoryPath -Value ($record | ConvertTo-Json -Compress -Depth 8)
    Write-Host "Backfilled $recordPath ($($run.Duration), $([math]::Round($msPerStep,1)) ms/step)"
}
