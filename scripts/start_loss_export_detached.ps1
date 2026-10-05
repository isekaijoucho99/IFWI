param(
    [Parameter(Mandatory = $true)][string]$SuiteDirectory,
    [string]$PythonExecutable,
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 1) { throw 'IntervalSeconds must be positive' }
$projectDirectory = Split-Path -Parent $PSScriptRoot
$suiteResolved = (Resolve-Path -LiteralPath $SuiteDirectory).Path
if (-not $PythonExecutable) {
    $PythonExecutable = (Get-Command python -CommandType Application -ErrorAction Stop).Source
}
$pythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path
$exportScript = Join-Path $PSScriptRoot 'plot_parameter_sweep_losses.py'
$taskName = 'IFWI_LossPlots_' + (Split-Path -Leaf $suiteResolved)
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.State -eq 'Running') {
        Write-Output ('Loss exporter is already running: ' + $taskName)
        return
    }
    throw ('Loss export task already exists; inspect it before relaunching: ' + $taskName)
}
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Project Python executable is missing' }
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$arguments = '-X utf8 -u "' + $exportScript + '" --suite "' + $suiteResolved + '" --watch --interval ' + $IntervalSeconds
$action = New-ScheduledTaskAction -Execute $pythonExecutable -Argument $arguments -WorkingDirectory $projectDirectory
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$null = Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings `
    -Description ('Read IFWI loss CSVs and export plots; no training or GPU work: ' + $suiteResolved)
[ordered]@{
    task_name = $taskName
    suite = $suiteResolved
    principal = $identity
    started_at = (Get-Date).ToString('o')
    interval_seconds = $IntervalSeconds
    trigger = 'manual; no recurring schedule'
    execution_time_limit = 'unlimited'
    arguments = $arguments
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $suiteResolved 'loss-export-launch.json') -Encoding UTF8
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $suiteResolved 'loss-export-task.xml') -Encoding UTF8
Start-ScheduledTask -TaskName $taskName
Write-Output ('Started loss exporter: ' + $taskName)
