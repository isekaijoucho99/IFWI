param(
    [Parameter(Mandatory = $true)][string]$SuiteDirectory
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$suiteResolved = (Resolve-Path -LiteralPath $SuiteDirectory).Path
$monitorScript = Join-Path $PSScriptRoot 'watch_parameter_sweep.ps1'
$powershellExecutable = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$taskName = 'IFWI_ProgressMonitor_' + (Split-Path -Leaf $suiteResolved)
$arguments = '-NoLogo -NoProfile -NoExit -WindowStyle Normal -ExecutionPolicy Bypass -File "' + $monitorScript + '" -SuiteDirectory "' + $suiteResolved + '"'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {
    Write-Output ('Progress monitor is already running: ' + $taskName)
    return
}
if (-not $existing) {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $action = New-ScheduledTaskAction -Execute $powershellExecutable -Argument $arguments -WorkingDirectory $projectDirectory
    $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $null = Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings `
        -Description ('Visible read-only IFWI progress monitor, independent of Codex: ' + $suiteResolved)
}
[ordered]@{
    task_name = $taskName
    suite = $suiteResolved
    started_at = (Get-Date).ToString('o')
    trigger = 'manual; no recurring schedule'
    execution_time_limit = 'unlimited'
    purpose = 'visible read-only monitor; does not start or stop training'
    arguments = $arguments
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $suiteResolved 'monitor-launch.json') -Encoding UTF8
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $suiteResolved 'monitor-task.xml') -Encoding UTF8
Start-ScheduledTask -TaskName $taskName
Write-Output ('Opened independent progress monitor: ' + $taskName)
