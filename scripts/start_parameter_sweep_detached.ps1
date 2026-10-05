param(
    [Parameter(Mandatory = $true)][string]$SuiteDirectory,
    [string]$PythonExecutable,
    [string]$Device = 'cuda:0',
    [switch]$OpenMonitor
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$suiteDirectoryResolved = (Resolve-Path -LiteralPath $SuiteDirectory).Path
if (-not $PythonExecutable) {
    $PythonExecutable = (Get-Command python -CommandType Application -ErrorAction Stop).Source
}
$pythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path
$recoveryScript = Join-Path $PSScriptRoot 'resume_parameter_sweep.py'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Project Python executable is missing' }

# This is a manually started OS task with no recurring trigger.
$taskName = 'IFWI_ParameterSweep_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'Task name already exists; refusing to overwrite'
}
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$arguments = '-X utf8 -u "' + $recoveryScript + '" --suite "' + $suiteDirectoryResolved + '" --device "' + $Device + '"'
$action = New-ScheduledTaskAction -Execute $pythonExecutable -Argument $arguments -WorkingDirectory $projectDirectory
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$null = Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings `
    -Description ('Continue the saved IFWI parameter sweep independently of the chat app: ' + $suiteDirectoryResolved)

$launchRecord = [ordered]@{
    task_name = $taskName
    task_path = '\'
    suite = $suiteDirectoryResolved
    principal = $identity
    started_at = (Get-Date).ToString('o')
    trigger = 'manual; no recurring schedule'
    execution_time_limit = 'unlimited'
    python = $pythonExecutable
    arguments = $arguments
}
$launchRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $suiteDirectoryResolved 'detached-launch.json') -Encoding UTF8
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $suiteDirectoryResolved 'detached-task.xml') -Encoding UTF8
Start-ScheduledTask -TaskName $taskName
Write-Output ('Started independent Windows task: ' + $taskName)

if ($OpenMonitor) {
    & (Join-Path $PSScriptRoot 'start_progress_monitor_detached.ps1') -SuiteDirectory $suiteDirectoryResolved
}
