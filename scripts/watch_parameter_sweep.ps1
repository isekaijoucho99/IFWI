param(
    [Parameter(Mandatory = $true)][string]$SuiteDirectory,
    [int]$RunnerProcessId = 0
)

$Host.UI.RawUI.WindowTitle = 'IFWI - ' + (Split-Path -Leaf $SuiteDirectory)
$ErrorActionPreference = 'Stop'
$seenLines = @{}
$lastState = ''
Write-Host 'IFWI: one-variable parameter experiments; see manifest.json for seed and update budget' -ForegroundColor Cyan
Write-Host "Results: $SuiteDirectory"
Write-Host 'This window monitors training. Closing it does not stop the training process.'
Write-Host 'Preparing observations can take several minutes before Update 1 appears.'
while ($true) {
    $statePath = Join-Path $SuiteDirectory 'active.json'
    if (Test-Path -LiteralPath $statePath) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
            $stateLabel = "$($state.state) | $($state.name) | completed $($state.completed_runs)/$($state.total_runs)"
            if ($stateLabel -ne $lastState) {
                Write-Host "`n=== $stateLabel ===" -ForegroundColor Cyan
                $lastState = $stateLabel
            }
            # Drain every log so the final lines of a previous experiment are retained.
            $logsPath = Join-Path $SuiteDirectory 'logs'
            if (Test-Path -LiteralPath $logsPath) {
                foreach ($logFile in (Get-ChildItem -LiteralPath $logsPath -Filter '*.log' | Sort-Object CreationTime)) {
                    $lines = @(Get-Content -LiteralPath $logFile.FullName -Encoding UTF8)
                    $offset = 0
                    if ($seenLines.ContainsKey($logFile.FullName)) { $offset = $seenLines[$logFile.FullName] }
                    if ($lines.Count -lt $offset) { $offset = 0 }
                    for ($i = $offset; $i -lt $lines.Count; $i++) {
                        Write-Host "[$($logFile.BaseName)] $($lines[$i])"
                    }
                    $seenLines[$logFile.FullName] = $lines.Count
                }
            }
            if ($state.state -ne 'running') {
                Write-Host "Suite finished with state: $($state.state)" -ForegroundColor Yellow
                if ($state.error) { Write-Host $state.error -ForegroundColor Red }
                break
            }
        } catch {
            Write-Host "Waiting for readable status/log: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    if ($RunnerProcessId -gt 0 -and -not (Get-Process -Id $RunnerProcessId -ErrorAction SilentlyContinue)) {
        Write-Host 'Runner exited. Check active.json and the console/error logs beside the suite directory.' -ForegroundColor Yellow
        break
    }
    Start-Sleep -Seconds 2
}
