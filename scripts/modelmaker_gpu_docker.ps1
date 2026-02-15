param(
  [Parameter(Position = 0)]
  [ValidateSet("build", "gpu-check", "run")]
  [string]$Action = "gpu-check",

  [string]$Image = "owli-modelmaker-gpu:tf2.8.4",

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$dockerfile = Join-Path $repoRoot "docker/modelmaker-gpu/Dockerfile"

function Invoke-Docker {
  param([string[]]$Command)
  & docker @Command
}

switch ($Action) {
  "build" {
    Invoke-Docker @("build", "-f", $dockerfile, "-t", $Image, $repoRoot)
  }
  "gpu-check" {
    Invoke-Docker @(
      "run", "--rm", "--gpus", "all",
      "-v", "${repoRoot}:/workspace",
      "-w", "/workspace",
      $Image,
      "python3.9", "-c",
      "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
    )
  }
  "run" {
    if (-not $Args -or $Args.Count -eq 0) {
      throw "Missing owli_train args. Example: .\scripts\modelmaker_gpu_docker.ps1 run train efficientdet configs\efficientdet_lite2_coco2017.yaml --max-steps 500 --subset-seed 1337 --require-gpu"
    }
    $cmd = @(
      "run", "--rm", "--gpus", "all",
      "-v", "${repoRoot}:/workspace",
      "-w", "/workspace",
      $Image,
      "python3.9", "-m", "owli_train"
    ) + $Args
    Invoke-Docker $cmd
  }
}
