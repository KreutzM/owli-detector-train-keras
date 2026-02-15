[CmdletBinding()]
param(
    [string]$CocoRoot = "data\coco2017",
    [switch]$WithBaseline,
    [string]$BaselineOut = "work\models\efficientdet_lite2_baseline.tflite",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$valZipUrl = "http://images.cocodataset.org/zips/val2017.zip"
$annZipUrl = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
$baselineUrl = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite2_detection_metadata_1.tflite"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $valZipPath = Join-Path $CocoRoot "val2017.zip"
    $annZipPath = Join-Path $CocoRoot "annotations_trainval2017.zip"
    $valDir = Join-Path $CocoRoot "val2017"
    $annFile = Join-Path $CocoRoot "annotations\instances_val2017.json"

    New-Item -ItemType Directory -Path $CocoRoot -Force | Out-Null

    if ($Force -and (Test-Path -Path $valDir)) {
        Remove-Item -Path $valDir -Recurse -Force
    }
    if ($Force -and (Test-Path -Path $annFile)) {
        Remove-Item -Path $annFile -Force
    }

    if ($Force -or -not (Test-Path -Path $valZipPath -PathType Leaf)) {
        Write-Host ">> Downloading val2017.zip"
        Invoke-WebRequest -Uri $valZipUrl -OutFile $valZipPath
    } else {
        Write-Host ">> Using cached: $valZipPath"
    }

    $hasValImages = $false
    if (Test-Path -Path $valDir -PathType Container) {
        $hasValImages = @(Get-ChildItem -Path $valDir -Filter *.jpg -File -ErrorAction SilentlyContinue).Count -gt 0
    }
    if (-not $hasValImages) {
        Write-Host ">> Extracting val2017 images"
        Expand-Archive -Path $valZipPath -DestinationPath $CocoRoot -Force
    }

    if ($Force -or -not (Test-Path -Path $annZipPath -PathType Leaf)) {
        Write-Host ">> Downloading annotations_trainval2017.zip"
        Invoke-WebRequest -Uri $annZipUrl -OutFile $annZipPath
    } else {
        Write-Host ">> Using cached: $annZipPath"
    }

    if (-not (Test-Path -Path $annFile -PathType Leaf)) {
        Write-Host ">> Extracting COCO annotations"
        Expand-Archive -Path $annZipPath -DestinationPath $CocoRoot -Force
    }

    if ($WithBaseline) {
        $baselineDir = Split-Path -Parent $BaselineOut
        if ($baselineDir) {
            New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
        }
        if ($Force -or -not (Test-Path -Path $BaselineOut -PathType Leaf)) {
            Write-Host ">> Downloading baseline model"
            Invoke-WebRequest -Uri $baselineUrl -OutFile $BaselineOut
        } else {
            Write-Host ">> Using cached: $BaselineOut"
        }
    }

    if (-not (Test-Path -Path $annFile -PathType Leaf)) {
        throw "Missing annotations after setup: $annFile"
    }
    if (-not (Test-Path -Path $valDir -PathType Container)) {
        throw "Missing val image directory after setup: $valDir"
    }

    $jpgCount = @(Get-ChildItem -Path $valDir -Filter *.jpg -File -ErrorAction SilentlyContinue).Count
    Write-Host "OK coco_root=$CocoRoot"
    Write-Host "OK images_dir=$valDir (jpg=$jpgCount)"
    Write-Host "OK annotations=$annFile"
    if ($WithBaseline) {
        Write-Host "OK baseline_model=$BaselineOut"
    }
}
finally {
    Pop-Location
}
