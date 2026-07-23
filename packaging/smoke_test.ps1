[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExeDir,

    [Parameter(Mandatory = $true)]
    [string]$Workspace,

    [string]$ReportPath = "",

    [string]$RuntimeDir = "",

    [int]$GuiTimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$RequestedWorkspace = [IO.Path]::GetFullPath($Workspace)
$ProtectedRoots = @(
    (Join-Path $ProjectRoot "dataset"),
    (Join-Path $ProjectRoot "models"),
    (Join-Path $ProjectRoot "runs"),
    (Join-Path $ProjectRoot "logs")
)
foreach ($ProtectedRoot in $ProtectedRoots) {
    $FullProtected = [IO.Path]::GetFullPath($ProtectedRoot)
    if ($RequestedWorkspace -eq $FullProtected -or $RequestedWorkspace.StartsWith($FullProtected + [IO.Path]::DirectorySeparatorChar)) {
        throw "Smoke workspace must not be inside a real runtime directory: $FullProtected"
    }
}

$SessionRoot = Join-Path $RequestedWorkspace ("smoke-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + $PID)
$SmokeWorkspace = Join-Path $SessionRoot "workspace"
$SmokeState = Join-Path $SessionRoot "state"
$SmokeYoloConfig = Join-Path $SmokeState "ultralytics"
$SmokeDataset = Join-Path $SmokeWorkspace "dataset"
$SmokeRuns = Join-Path $SmokeWorkspace "runs"
New-Item -ItemType Directory -Force `
    $SmokeState, `
    $SmokeYoloConfig, `
    $SmokeRuns, `
    (Join-Path $SmokeDataset "images\train"), `
    (Join-Path $SmokeDataset "labels\train") | Out-Null
Set-Content -LiteralPath (Join-Path $SmokeDataset "classes.txt") -Value "" -Encoding UTF8
Set-Content -LiteralPath (Join-Path $SmokeDataset "data.yaml") -Encoding UTF8 -Value @(
    "path: .",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "names: []"
)

$Candidate = [IO.Path]::GetFullPath($ExeDir)
if (-not (Test-Path -LiteralPath (Join-Path $Candidate "YOLO工具箱.exe"))) {
    $Nested = Join-Path $Candidate "YOLO数据标注工具箱"
    if (Test-Path -LiteralPath (Join-Path $Nested "YOLO工具箱.exe")) {
        $Candidate = $Nested
    }
}
$GuiExe = Join-Path $Candidate "YOLO工具箱.exe"
$WorkerExe = ""
$RuntimeCandidate = ""
$Results = @()

function Add-SmokeResult {
    param(
        [string]$Name,
        [ValidateSet("PASS", "FAIL", "SKIP")]
        [string]$Status,
        [string]$Details
    )
    $script:Results += [pscustomobject]@{
        Name = $Name
        Status = $Status
        Details = $Details
    }
}

function ConvertTo-CommandArgument {
    param([string]$Value)
    if ($null -eq $Value) {
        return '""'
    }
    return '"' + $Value.Replace('"', '\"') + '"'
}

function New-SmokeProcessInfo {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [bool]$RedirectOutput
    )
    $Info = New-Object System.Diagnostics.ProcessStartInfo
    $Info.FileName = $FilePath
    $Info.Arguments = (($Arguments | ForEach-Object { ConvertTo-CommandArgument $_ }) -join " ")
    $Info.UseShellExecute = $false
    $Info.CreateNoWindow = $true
    $Info.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $Info.RedirectStandardOutput = $RedirectOutput
    $Info.RedirectStandardError = $RedirectOutput
    $Info.EnvironmentVariables["WORKBUDDY_STATE_DIR"] = $SmokeState
    $Info.EnvironmentVariables["WORKBUDDY_WORKSPACE_DIR"] = $SmokeWorkspace
    $Info.EnvironmentVariables["YOLO_CONFIG_DIR"] = $SmokeYoloConfig
    $Info.EnvironmentVariables["PATH"] = "$env:SystemRoot\System32;$env:SystemRoot"
    $Info.EnvironmentVariables.Remove("PYTHONHOME")
    $Info.EnvironmentVariables.Remove("PYTHONPATH")
    return $Info
}

function Invoke-SmokeWorker {
    param(
        [string[]]$Arguments,
        [int]$TimeoutSeconds = 60
    )
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = New-SmokeProcessInfo $WorkerExe $Arguments $true
    $null = $Process.Start()
    $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
    $StderrTask = $Process.StandardError.ReadToEndAsync()
    if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) {
        $Process.Kill()
        $Process.WaitForExit()
        return [pscustomobject]@{
            ExitCode = -999
            TimedOut = $true
            Output = $StdoutTask.Result + "`n" + $StderrTask.Result
        }
    }
    $Process.WaitForExit()
    return [pscustomobject]@{
        ExitCode = $Process.ExitCode
        TimedOut = $false
        Output = $StdoutTask.Result + "`n" + $StderrTask.Result
    }
}

$BaseLayoutOk = (Test-Path -LiteralPath $GuiExe) -and `
    (Test-Path -LiteralPath (Join-Path $Candidate "_internal\PyQt5\Qt5\plugins\platforms\qwindows.dll")) -and `
    (Test-Path -LiteralPath (Join-Path $Candidate "_internal\runtime_catalog.json")) -and `
    (-not (Test-Path -LiteralPath (Join-Path $Candidate "YOLO工具箱Worker.exe"))) -and `
    (-not (Test-Path -LiteralPath (Join-Path $Candidate "_internal\torch")))
Add-SmokeResult "Base portable layout" $(if ($BaseLayoutOk) { "PASS" } else { "FAIL" }) $Candidate
if (-not $BaseLayoutOk) {
    throw "Base portable EXE layout is incomplete or contains ML runtime files: $Candidate"
}

$GuiHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $GuiExe).Hash.ToLowerInvariant()
$WorkerHash = "not-tested"
$WorkerAvailable = $false
if ($RuntimeDir) {
    $RuntimeCandidate = [IO.Path]::GetFullPath($RuntimeDir)
    $WorkerExe = Join-Path $RuntimeCandidate "YOLO工具箱Worker.exe"
    $RuntimeLayoutOk = (Test-Path -LiteralPath $WorkerExe) -and `
        (Test-Path -LiteralPath (Join-Path $RuntimeCandidate "runtime.json")) -and `
        (Test-Path -LiteralPath (Join-Path $RuntimeCandidate "_internal\torch\lib\torch_cpu.dll")) -and `
        (-not (Test-Path -LiteralPath (Join-Path $RuntimeCandidate "YOLO工具箱.exe")))
    Add-SmokeResult "Worker runtime layout" $(if ($RuntimeLayoutOk) { "PASS" } else { "FAIL" }) $RuntimeCandidate
    if (-not $RuntimeLayoutOk) {
        throw "Worker runtime layout is incomplete: $RuntimeCandidate"
    }
    $WorkerAvailable = $true
    $WorkerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $WorkerExe).Hash.ToLowerInvariant()
}

if ($WorkerAvailable) {
$Unknown = Invoke-SmokeWorker @("bogus") 30
$UnknownOk = $Unknown.ExitCode -eq 2 -and $Unknown.Output.Contains("__YOLO_WORKER_ERROR__")
Add-SmokeResult "Worker unknown command" $(if ($UnknownOk) { "PASS" } else { "FAIL" }) ("exit=" + $Unknown.ExitCode)

$HelpFailures = @()
foreach ($Kind in @("train", "detect", "evaluate", "tensorboard")) {
    $Help = Invoke-SmokeWorker @($Kind, "--help") 30
    if ($Help.ExitCode -ne 0) {
        $HelpFailures += "$Kind=$($Help.ExitCode)"
    }
}
Add-SmokeResult "Worker command parsers" $(if ($HelpFailures.Count -eq 0) { "PASS" } else { "FAIL" }) $(if ($HelpFailures.Count -eq 0) { "all exit=0" } else { $HelpFailures -join ", " })

$MissingModel = Join-Path $SessionRoot "missing-model.pt"
$MissingImage = Join-Path $SessionRoot "missing-image.jpg"
$MissingOutput = Join-Path $SessionRoot "missing-result.json"
$Detect = Invoke-SmokeWorker @(
    "detect",
    "--model", $MissingModel,
    "--image", $MissingImage,
    "--output", $MissingOutput
) 120
$MISSING_MODEL_EXIT = $Detect.ExitCode
$DetectOk = $MISSING_MODEL_EXIT -eq 1 -and `
    $Detect.Output.Contains("__YOLO_TASK_EVENT__") -and `
    $Detect.Output.Contains('"type":"failed"')
Add-SmokeResult "Torch and Ultralytics import path" $(if ($DetectOk) { "PASS" } else { "FAIL" }) ("MISSING_MODEL_EXIT=" + $MISSING_MODEL_EXIT)

$TensorboardProcess = New-Object System.Diagnostics.Process
$TensorboardProcess.StartInfo = New-SmokeProcessInfo $WorkerExe @(
    "tensorboard",
    "--logdir", $SmokeRuns,
    "--host", "127.0.0.1",
    "--port", "auto"
) $true
$null = $TensorboardProcess.Start()
$TensorboardStdout = $TensorboardProcess.StandardOutput.ReadToEndAsync()
$TensorboardStderr = $TensorboardProcess.StandardError.ReadToEndAsync()
Start-Sleep -Seconds 12
$TensorboardAlive = -not $TensorboardProcess.HasExited
if ($TensorboardAlive) {
    $TensorboardProcess.Kill()
}
$TensorboardProcess.WaitForExit()
$TensorboardText = $TensorboardStdout.Result + "`n" + $TensorboardStderr.Result
$TensorboardReady = $TensorboardText.Contains("__YOLO_TASK_EVENT__") -and `
    $TensorboardText.Contains('"type":"result"') -and `
    $TensorboardText.Contains("http://127.0.0.1:")
Add-SmokeResult "TensorBoard embedded server" $(if ($TensorboardReady) { "PASS" } else { "FAIL" }) $(if ($TensorboardReady) { "ready event observed" } else { "ready event missing" })
} else {
    Add-SmokeResult "Worker runtime layout" "SKIP" "RuntimeDir was not provided."
    Add-SmokeResult "Worker command parsers" "SKIP" "RuntimeDir was not provided."
    Add-SmokeResult "Torch and Ultralytics import path" "SKIP" "RuntimeDir was not provided."
    Add-SmokeResult "TensorBoard embedded server" "SKIP" "RuntimeDir was not provided."
}

$GuiProcess = New-Object System.Diagnostics.Process
$GuiProcess.StartInfo = New-SmokeProcessInfo $GuiExe @() $false
$null = $GuiProcess.Start()
$GuiReady = $false
$Deadline = [DateTime]::UtcNow.AddSeconds($GuiTimeoutSeconds)
while ([DateTime]::UtcNow -lt $Deadline -and -not $GuiProcess.HasExited) {
    Start-Sleep -Milliseconds 250
    $GuiProcess.Refresh()
    if ($GuiProcess.MainWindowTitle -eq "YOLO 数据标注工具箱") {
        $GuiReady = $true
        break
    }
}
$CloseRequested = $false
if (-not $GuiProcess.HasExited) {
    $CloseRequested = $GuiProcess.CloseMainWindow()
    if ($CloseRequested) {
        $null = $GuiProcess.WaitForExit(5000)
    }
}
if (-not $GuiProcess.HasExited) {
    $GuiProcess.Kill()
    $GuiProcess.WaitForExit()
}
$GuiOk = $GuiReady -and $GuiProcess.ExitCode -eq 0
Add-SmokeResult "GUI launch without Python PATH" $(if ($GuiOk) { "PASS" } else { "FAIL" }) ("ready=$GuiReady exit=$($GuiProcess.ExitCode) close=$CloseRequested")

Start-Sleep -Seconds 1
$ProcessRoots = @($Candidate)
if ($RuntimeCandidate) {
    $ProcessRoots += $RuntimeCandidate
}
$RemainingCandidateProcesses = @(
    Get-CimInstance Win32_Process | Where-Object {
        $ExecutablePath = $_.ExecutablePath
        $ExecutablePath -and @($ProcessRoots | Where-Object {
            $ExecutablePath.StartsWith($_, [StringComparison]::OrdinalIgnoreCase)
        }).Count -gt 0
    }
).Count
Add-SmokeResult "RemainingCandidateProcesses" $(if ($RemainingCandidateProcesses -eq 0) { "PASS" } else { "FAIL" }) ("count=" + $RemainingCandidateProcesses)

$PartFiles = @(Get-ChildItem -Recurse -File $SessionRoot -Filter "*.part" -ErrorAction SilentlyContinue).Count
Add-SmokeResult "Partial download cleanup" $(if ($PartFiles -eq 0) { "PASS" } else { "FAIL" }) ("part files=" + $PartFiles)

Add-SmokeResult "GPU inference" "SKIP" "No model or user image is included in the isolated smoke workspace."
Add-SmokeResult "Training cancellation" "SKIP" "Requires an explicit disposable model and dataset fixture."
Add-SmokeResult "Model download cancellation" "SKIP" "Covered by unit tests; network download is not started by smoke test."
Add-SmokeResult "Installer upgrade and uninstall" "SKIP" "Validated separately with the Inno Setup candidate."

if (-not $ReportPath) {
    $ReportPath = Join-Path $SessionRoot "smoke-report.md"
}
$FullReportPath = [IO.Path]::GetFullPath($ReportPath)
$ReportDirectory = Split-Path $FullReportPath -Parent
if ($ReportDirectory) {
    New-Item -ItemType Directory -Force $ReportDirectory | Out-Null
}
$OsCaption = (Get-CimInstance Win32_OperatingSystem).Caption
$Lines = @(
    "# Windows EXE Smoke Test",
    "",
    "- Generated: $([DateTime]::UtcNow.ToString('o'))",
    "- OS: $OsCaption",
    "- Candidate: $Candidate",
    "- Runtime candidate: $(if ($RuntimeCandidate) { $RuntimeCandidate } else { 'not-tested' })",
    "- Session workspace: $SessionRoot",
    "- GUI SHA256: $GuiHash",
    "- Worker SHA256: $WorkerHash",
    "",
    "| Check | Status | Details |",
    "| --- | --- | --- |"
)
foreach ($Result in $Results) {
    $SafeDetails = [string]$Result.Details
    $SafeDetails = $SafeDetails.Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
    $Lines += "| $($Result.Name) | $($Result.Status) | $SafeDetails |"
}
Set-Content -LiteralPath $FullReportPath -Value $Lines -Encoding UTF8
$JsonPath = [IO.Path]::ChangeExtension($FullReportPath, ".json")
$Results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $JsonPath -Encoding UTF8

$Failed = @($Results | Where-Object { $_.Status -eq "FAIL" }).Count
Write-Output "Report: $FullReportPath"
Write-Output "JSON: $JsonPath"
Write-Output "PASS=$(@($Results | Where-Object { $_.Status -eq 'PASS' }).Count) FAIL=$Failed SKIP=$(@($Results | Where-Object { $_.Status -eq 'SKIP' }).Count)"
if ($Failed -gt 0) {
    exit 1
}
exit 0

