[CmdletBinding()]
param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    Write-Host ">> $PythonExe $($Args -join ' ')"
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit=$LASTEXITCODE): $PythonExe $($Args -join ' ')"
    }
}

function Resolve-Coco128Root {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExtractRoot
    )

    $candidates = @($ExtractRoot)
    $candidates += Get-ChildItem -Path $ExtractRoot -Directory -Recurse | ForEach-Object {
        $_.FullName
    }
    foreach ($candidate in $candidates) {
        $imagesPath = Join-Path $candidate "images"
        $labelsPath = Join-Path $candidate "labels"
        if ((Test-Path -Path $imagesPath -PathType Container) -and (Test-Path -Path $labelsPath -PathType Container)) {
            return $candidate
        }
    }

    throw "Could not locate COCO128 YOLO root under: $ExtractRoot"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $dataRoot = Join-Path $repoRoot "data"
    $workRoot = Join-Path $repoRoot "work"
    New-Item -ItemType Directory -Path $dataRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $workRoot -Force | Out-Null

    $zipUrl = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    $zipPath = Join-Path $dataRoot "coco128.zip"
    $extractRoot = Join-Path $dataRoot "coco128_extract"
    $datasetRoot = Join-Path $dataRoot "coco128"

    if (-not (Test-Path -Path $zipPath -PathType Leaf)) {
        Write-Host "Downloading COCO128 from Ultralytics assets..."
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
    } else {
        Write-Host "Using cached archive: $zipPath"
    }

    if (Test-Path -Path $extractRoot) {
        Remove-Item -Path $extractRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $extractRoot -Force | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath $extractRoot -Force

    $resolvedExtractedRoot = Resolve-Coco128Root -ExtractRoot $extractRoot
    if (Test-Path -Path $datasetRoot) {
        Remove-Item -Path $datasetRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $datasetRoot -Force | Out-Null
    Copy-Item -Path (Join-Path $resolvedExtractedRoot "*") -Destination $datasetRoot -Recurse -Force

    & $PythonExe -c "import tflite_model_maker" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Missing TensorFlow Lite Model Maker dependencies."
        Write-Host "Install with: pip install -r requirements\modelmaker.txt"
        exit 1
    }

    $cocoOut = "work\datasets\coco128\instances.json"
    $splitsDir = "work\datasets\coco128\splits"
    $splitsJson = "work\datasets\coco128\splits\splits.json"
    $csvOut = "work\datasets\coco128\modelmaker.csv"

    Invoke-Python -Args @(
        "-m", "owli_train", "dataset", "import", "yolo",
        "--yolo-dir", "data\coco128",
        "--out", $cocoOut
    )
    Invoke-Python -Args @(
        "-m", "owli_train", "dataset", "split",
        "--coco", $cocoOut,
        "--out-dir", $splitsDir,
        "--seed", "1337"
    )
    Invoke-Python -Args @(
        "-m", "owli_train", "dataset", "export", "modelmaker-csv",
        "--coco", $cocoOut,
        "--images-dir", "data\coco128\images",
        "--splits-json", $splitsJson,
        "--out", $csvOut
    )

    $trainOutput = & $PythonExe -m owli_train train efficientdet --variant lite2 --config configs\efficientdet_lite2_coco128.yaml --max-steps 1 2>&1
    if ($LASTEXITCODE -ne 0) {
        $trainOutput | ForEach-Object { Write-Host $_ }
        throw "EfficientDet smoke training failed."
    }
    $trainOutput | ForEach-Object { Write-Host $_ }

    $runDirLine = $trainOutput | Where-Object { $_ -like "run_dir:*" } | Select-Object -Last 1
    if (-not $runDirLine) {
        throw "Could not parse run_dir from training output."
    }
    $runDir = $runDirLine.Split(":", 2)[1].Trim()

    $tflitePath = Join-Path $runDir "artifacts\model.tflite"
    if (-not (Test-Path -Path $tflitePath -PathType Leaf)) {
        throw "Expected TFLite artifact not found: $tflitePath"
    }

    Write-Host "Verified TFLite artifact: $tflitePath"
    Write-Host "Inspecting exported TFLite model..."
    Invoke-Python -Args @("-m", "owli_train", "inspect", "tflite", "--model", $tflitePath)

    Write-Host ""
    Write-Host "E2E smoke completed successfully."
    Write-Host "run_dir: $runDir"
    Write-Host "tflite: $tflitePath"
    Write-Host ""
    Write-Host "Generated paths:"
    Write-Host "  data\coco128.zip"
    Write-Host "  data\coco128\"
    Write-Host "  work\datasets\coco128\"
    Write-Host "  work\runs\<run_id>\"
    Write-Host ""
    Write-Host "Cleanup:"
    Write-Host "  Remove-Item -Path data\coco128, data\coco128_extract, work\datasets\coco128 -Recurse -Force"
    Write-Host "  Remove-Item -Path work\runs\<run_id> -Recurse -Force"
}
finally {
    Pop-Location
}
