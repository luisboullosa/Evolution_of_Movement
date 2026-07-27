# Summarize headless run timing from output/run_history.jsonl
param(
    [string]$HistoryPath = (Join-Path $PSScriptRoot "..\output\run_history.jsonl"),
    [string]$Architecture = "",
    [int]$MinSteps = 0
)

if (-not (Test-Path $HistoryPath)) {
    Write-Host "No run history at $HistoryPath"
    exit 0
}

$rows = @()
Get-Content $HistoryPath | ForEach-Object {
    if ([string]::IsNullOrWhiteSpace($_)) { return }
    try {
        $rows += ($_ | ConvertFrom-Json)
    } catch {
        Write-Warning "Skipping invalid JSON line: $_"
    }
}

if ($Architecture) {
    $rows = $rows | Where-Object { $_.sim.architecture_policy -eq $Architecture }
}
if ($MinSteps -gt 0) {
    $rows = $rows | Where-Object { $_.sim.steps_completed -ge $MinSteps }
}

if (-not $rows) {
    Write-Host "No matching runs."
    exit 0
}

function Format-Duration([double]$secs) {
    $ts = [TimeSpan]::FromSeconds($secs)
    return $ts.ToString("hh\:mm\:ss")
}

Write-Host "Run history: $HistoryPath"
Write-Host ""
Write-Host ("{0,-22} {1,8} {2,8} {3,10} {4,10} {5,12} {6}" -f `
    "Finished", "Steps", "Pop", "Duration", "ms/step", "Binary", "Policy")
Write-Host ("-" * 90)

foreach ($r in $rows) {
    $finished = [DateTimeOffset]::FromUnixTimeSeconds([int64]$r.timing.finished_at_unix_secs).LocalDateTime.ToString("yyyy-MM-dd HH:mm")
    $bin = $r.worker.binary.name
    $policy = $r.sim.architecture_policy
    $dur = Format-Duration $r.timing.duration_secs
    $ms = "{0:N1}" -f $r.timing.ms_per_step
    Write-Host ("{0,-22} {1,8} {2,8} {3,10} {4,10} {5,12} {6}" -f `
        $finished, $r.sim.steps_completed, $r.sim.final_population, $dur, $ms, $bin, $policy)
}

Write-Host ""
$avgMs = ($rows | ForEach-Object { [double]$_.timing.ms_per_step } | Measure-Object -Average).Average
$avgStepsPerSec = ($rows | ForEach-Object { [double]$_.timing.steps_per_sec } | Measure-Object -Average).Average
$fullRuns = @($rows | Where-Object { $_.sim.steps_completed -ge 1000 })
if ($fullRuns.Count -gt 0) {
    $avgMs = ($fullRuns | ForEach-Object { [double]$_.timing.ms_per_step } | Measure-Object -Average).Average
    $avgStepsPerSec = ($fullRuns | ForEach-Object { [double]$_.timing.steps_per_sec } | Measure-Object -Average).Average
    $rows = $fullRuns
}
Write-Host ("Average over {0} run(s): {1:N1} ms/step, {2:N1} steps/sec" -f $rows.Count, $avgMs, $avgStepsPerSec)

# Rough ETA helper for 100K steps at average ms/step
$eta100k = ($avgMs * 100000) / 1000.0 / 3600.0
Write-Host ("Rough 100K-step estimate at this average: {0:N1} hours per architecture" -f $eta100k)
