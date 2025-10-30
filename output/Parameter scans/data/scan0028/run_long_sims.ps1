# PowerShell script to run long evo-movement-sim.exe runs for selected parameter combos
# Created by generate_long_run_commands.py
Set-StrictMode -Version Latest
if (-not (Test-Path ".\build\Release\evo-movement-sim.exe")) { Write-Host "Warning: Release exe not found; ensure path is correct" }

Write-Host "No candidate parameter combos found; nothing to run."
