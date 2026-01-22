param(
    [Parameter(Mandatory = $false, Position = 0)]
    [string]$ConfigPath = "",

    [switch]$Release
)

$ErrorActionPreference = 'Stop'

# If no path is provided, default to run_config.json in the current directory.
if (-not $PSBoundParameters.ContainsKey('ConfigPath') -or [string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path (Get-Location).ProviderPath 'run_config.json'
}

function Find-RustRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StartDir
    )

    $dir = Resolve-Path $StartDir | Select-Object -ExpandProperty ProviderPath
    while ($true) {
        # Case 1: this directory is itself a Rust crate
        if (Test-Path (Join-Path $dir 'Cargo.toml')) {
            return $dir
        }

        # Case 2: this directory has a nested rust/ crate
        $nestedRust = Join-Path $dir 'rust'
        if (Test-Path (Join-Path $nestedRust 'Cargo.toml')) {
            return $nestedRust
        }

        $parent = Split-Path -Parent $dir
        if (-not $parent -or $parent -eq $dir) {
            break
        }
        $dir = $parent
    }

    throw "Could not find Cargo.toml starting from '$StartDir'"
}

# Resolve paths relative to the current working directory before changing location.
# IMPORTANT: do not create/overwrite config.json from this script.
if (-not (Test-Path $ConfigPath)) {
    $cwd = (Get-Location).ProviderPath
    $runCfgHere = Join-Path $cwd 'run_config.json'
    if (Test-Path $runCfgHere) {
        # Use run_config.json directly without writing a new config file.
        $ConfigFullPath = (Resolve-Path $runCfgHere).ProviderPath
    } else {
        Write-Error "Config file not found: $ConfigPath"
    }
} else {
    $ConfigFullPath = (Resolve-Path $ConfigPath).ProviderPath
}

# If config.json is provided but a sibling run_config.json exists, prefer run_config.json
$configDir = Split-Path -Parent $ConfigFullPath
$runCfgSibling = Join-Path $configDir 'run_config.json'
if ((Split-Path -Leaf $ConfigFullPath) -ieq 'config.json' -and (Test-Path $runCfgSibling)) {
    $ConfigFullPath = (Resolve-Path $runCfgSibling).ProviderPath
    $configDir = Split-Path -Parent $ConfigFullPath
}

function Get-NextRunDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SimDir
    )

    if (-not (Test-Path $SimDir)) {
        throw "SimDir not found: $SimDir"
    }

    $max = -1
    Get-ChildItem -Path $SimDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -match '^run(\d+)$') {
            $n = [int]$Matches[1]
            if ($n -gt $max) { $max = $n }
        }
    }

    $next = $max + 1
    $runName = ('run{0:0000}' -f $next)
    return (Join-Path $SimDir $runName)
}

# Find the Rust crate root regardless of where this script lives (scripts/, simXXXX, etc.)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rustDir = Find-RustRoot -StartDir $scriptDir

# If the config lives inside a simXXXX directory with a snapshot, generate
# webviewer outputs into simXXXX/runYYYY/ so they're traceable to the source population.
$snapshotPath = Join-Path $configDir 'population_snapshot.cbor'
$useSnapshot = Test-Path $snapshotPath

if ($useSnapshot) {
    # Normalize run_config.json so replay runs don't drop new population_snapshot.cbor
    # or generate_webviewer_outputs.ps1 inside runYYYY directories.
    $parentRunConfig = Join-Path $configDir 'run_config.json'
    $parentConfig = Join-Path $configDir 'config.json'
    if (Test-Path $parentConfig) {
        try {
            $j = Get-Content -LiteralPath $parentConfig -Raw | ConvertFrom-Json
            $j.burn_in = 0
            $j.steps = 100
            $j.write_outputs = $true
            $j.write_snapshot = $false
            $j.write_config = $false
            $j.write_frames = $false
            $j | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $parentRunConfig -Encoding utf8
        } catch {
            # If JSON conversion fails, fall back to just copying parent config.
            Copy-Item -LiteralPath $parentConfig -Destination $parentRunConfig -Force
        }
    }

    $runDir = Get-NextRunDir -SimDir $configDir
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $runDirFullPath = (Resolve-Path $runDir).ProviderPath
    $snapshotFullPath = (Resolve-Path $snapshotPath).ProviderPath

    # Copy configs into the run dir for traceability. These are the authoritative
    # evolved (config.json) and experiment (run_config.json) configs.
    if (Test-Path $parentConfig) {
        Copy-Item -LiteralPath $parentConfig -Destination (Join-Path $runDirFullPath 'config.json') -Force
    }
    if (Test-Path $parentRunConfig) {
        Copy-Item -LiteralPath $parentRunConfig -Destination (Join-Path $runDirFullPath 'run_config.json') -Force
    }
}

Push-Location $rustDir
try {
    if ($Release) {
        if ($useSnapshot) {
            cargo run --release --bin evo_movement_rust -- --config-file "$ConfigFullPath" --replay-snapshot "$snapshotFullPath" --output-run-dir "$runDirFullPath" --write-outputs --no-print-frames
        } else {
            cargo run --release --bin evo_movement_rust -- --config-file "$ConfigFullPath" --write-outputs --no-print-frames
        }
    } else {
        if ($useSnapshot) {
            cargo run --bin evo_movement_rust -- --config-file "$ConfigFullPath" --replay-snapshot "$snapshotFullPath" --output-run-dir "$runDirFullPath" --write-outputs --no-print-frames
        } else {
            cargo run --bin evo_movement_rust -- --config-file "$ConfigFullPath" --write-outputs --no-print-frames
        }
    }
}
finally {
    Pop-Location
}
