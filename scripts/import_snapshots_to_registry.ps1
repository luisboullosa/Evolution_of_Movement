# Import legacy output/*/snapshot.cbor files into output/snapshots/ (flat registry).
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$rustDir = Join-Path $repoRoot "rust"
Push-Location $rustDir
try {
    cargo run --release --bin evo_snapshot_import -- @args
} finally {
    Pop-Location
}
