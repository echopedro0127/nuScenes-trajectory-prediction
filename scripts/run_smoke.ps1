[CmdletBinding()]
param(
    [switch]$SkipTraining,
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$DataConfig = "configs\data\mini.yaml",
    [string]$OutputRoot = "outputs\mini"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = if ([IO.Path]::IsPathRooted($Python)) { $Python } else { Join-Path $root $Python }
if (-not (Test-Path $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
Set-Location $root

function Invoke-Python {
    param([string[]]$Arguments)
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Project-Path {
    param([string]$Value)
    if ([IO.Path]::IsPathRooted($Value)) { return $Value }
    return (Join-Path $root $Value)
}

$outputRootPath = Project-Path $OutputRoot
$gruDir = Join-Path $outputRootPath "gru"
$transformerDir = Join-Path $outputRootPath "transformer"
Invoke-Python @("scripts\prepare_data.py", "--config", $DataConfig)
if (-not $SkipTraining) {
    Invoke-Python @("scripts\train.py", "--data-config", $DataConfig, "--model-config", "configs\models\gru.yaml", "--train-split", "mini_train", "--val-split", "mini_val", "--history-seconds", "2.0", "--output-dir", $gruDir)
    Invoke-Python @("scripts\train.py", "--data-config", $DataConfig, "--model-config", "configs\models\transformer.yaml", "--train-split", "mini_train", "--val-split", "mini_val", "--history-seconds", "2.0", "--output-dir", $transformerDir)
} elseif (-not (Test-Path (Join-Path $gruDir "best.pt")) -or -not (Test-Path (Join-Path $transformerDir "best.pt"))) {
    throw "-SkipTraining requires best.pt in both $gruDir and $transformerDir"
}

Invoke-Python @("scripts\evaluate.py", "--data-config", $DataConfig, "--split", "mini_val", "--model-config", "configs\models\cv.yaml", "--output-dir", (Join-Path $outputRootPath "cv"))
Invoke-Python @("scripts\evaluate.py", "--data-config", $DataConfig, "--split", "mini_val", "--model-config", "configs\models\cv_heading.yaml", "--output-dir", (Join-Path $outputRootPath "cv_heading"))
Invoke-Python @("scripts\evaluate.py", "--data-config", $DataConfig, "--split", "mini_val", "--model-config", "configs\models\gru.yaml", "--checkpoint", (Join-Path $gruDir "best.pt"), "--output-dir", (Join-Path $outputRootPath "gru_eval"))
Invoke-Python @("scripts\evaluate.py", "--data-config", $DataConfig, "--split", "mini_val", "--model-config", "configs\models\transformer.yaml", "--checkpoint", (Join-Path $transformerDir "best.pt"), "--output-dir", (Join-Path $outputRootPath "transformer_eval"))

$basePredictions = Join-Path $outputRootPath "gru_eval\predictions.npz"
$correctorDir = Join-Path $outputRootPath "corrector"
if (-not $SkipTraining) {
    Invoke-Python @("scripts\train_corrector.py", "--data-config", $DataConfig, "--split", "mini_val", "--base-predictions", $basePredictions, "--output-dir", $correctorDir)
} elseif (-not (Test-Path (Join-Path $correctorDir "best.pt"))) {
    throw "-SkipTraining requires best.pt in $correctorDir"
}
Invoke-Python @("scripts\evaluate_online.py", "--data-config", $DataConfig, "--split", "mini_val", "--base-predictions", $basePredictions, "--corrector-checkpoint", (Join-Path $correctorDir "best.pt"), "--output-dir", (Join-Path $outputRootPath "online"))
Invoke-Python @("scripts\analyze_maneuvers.py", "--data-config", $DataConfig, "--split", "mini_val", "--evaluation-dir", (Join-Path $outputRootPath "gru_eval"), "--output-dir", (Join-Path $outputRootPath "analysis"))
