# Run a headless sim binary with Write-Progress driven by [progress] lines (piped stdout).
function Invoke-HeadlessSimWithProgress {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Activity,
        [string[]]$ExtraArgs = @()
    )

    $exeArgs = @("--config-file=$Config") + $ExtraArgs
    & $Exe @exeArgs 2>&1 | ForEach-Object {
        $line = "$_"
        if ($line -match '^\[progress\]\s+\[.*?\]\s+(\d+)%') {
            $pct = [Math]::Min(100, [int]$Matches[1])
            $status = $line -replace '^\[progress\]\s+', ''
            Write-Progress -Activity $Activity -Status $status -PercentComplete $pct
        } else {
            Write-Host $line
        }
    }
    Write-Progress -Activity $Activity -Completed
    return $LASTEXITCODE
}
