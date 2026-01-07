param(
    [string]$RemoteHost = "192.168.2.129",
    [string]$SshUser = "root",
    [string]$RemoteRepoDir = "/root/projects/Movement",
    [string]$RemoteRunBase = "/root/projects/scan_runs",

    [int]$Sample = 200,
    [int]$Workers = 1,
    [int]$Steps = 200,
    [int]$InitialPopulation = 200,
    [int]$MaxPopulation = 1000,
    [int]$Threshold = 200,
    [int]$Seed = 0,

    [switch]$Deep,
    [int]$DeepSteps = 10000,
    [int]$DeepMaxPopulation = 20000,
    [int]$MaxDeep = 25,

    [switch]$ForceDeployHelper,
    [switch]$DebugBuild,

    [switch]$Bootstrap,
    [switch]$BootstrapBuild,
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$orchestrator = Join-Path $repoRoot "scripts\remote_scan_orchestrator.py"

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) {
    $py = "python"
}

$argsList = @(
    $orchestrator,
    "--remote-host", $RemoteHost,
    "--ssh-user", $SshUser,
    "--remote-repo-dir", $RemoteRepoDir,
    "--remote-run-base", $RemoteRunBase,
    "--sample", $Sample,
    "--workers", $Workers,
    "--steps", $Steps,
    "--initial-population", $InitialPopulation,
    "--max-population", $MaxPopulation,
    "--threshold", $Threshold
)

if ($Seed -ne 0) {
    $argsList += "--seed"
    $argsList += $Seed
}

if ($Bootstrap) {
    $argsList += "--bootstrap"
    $argsList += "--local-repo-dir"
    $argsList += $repoRoot
}

if ($BootstrapBuild) {
    $argsList += "--bootstrap-build"
}

if ($Deep) {
    $argsList += "--deep"
    $argsList += "--deep-steps"
    $argsList += $DeepSteps
    $argsList += "--deep-max-population"
    $argsList += $DeepMaxPopulation
    $argsList += "--max-deep"
    $argsList += $MaxDeep
}

if ($ForceDeployHelper) {
    $argsList += "--force-deploy-helper"
}

if (-not $DebugBuild) {
    # Orchestrator defaults to --release=true; keep explicit.
    $argsList += "--release"
}

if ($null -ne $ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $argsList += $ExtraArgs
}

& $py @argsList
exit $LASTEXITCODE
