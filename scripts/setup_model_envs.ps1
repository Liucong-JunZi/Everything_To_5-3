param(
    [ValidateSet("all", "dots", "lighton", "paddle-tf", "paddle-paddle", "zhen", "layout")]
    [string]$Target = "all",
    [switch]$CreateOnly,
    [string]$PythonExe = "D:\ancoda\python.exe",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
if (!(Test-Path $env:UV_CACHE_DIR)) {
    New-Item -ItemType Directory -Path $env:UV_CACHE_DIR | Out-Null
}

$specs = @{
    "dots" = @{
        EnvPath = ".venv-dots-ocr"
        ReqPath = "deployment/requirements/dots-ocr.txt"
    }
    "lighton" = @{
        EnvPath = ".venv-lighton-ocr"
        ReqPath = "deployment/requirements/lighton-ocr.txt"
    }
    "paddle-tf" = @{
        EnvPath = ".venv-paddleocr-vl-tf"
        ReqPath = "deployment/requirements/paddleocr-vl-transformers.txt"
    }
    "paddle-paddle" = @{
        EnvPath = ".venv-paddleocr-vl-paddle"
        ReqPath = "deployment/requirements/paddleocr-vl-paddle.txt"
    }
    "zhen" = @{
        EnvPath = ".venv-zhen-latex-ocr"
        ReqPath = "deployment/requirements/zhen-latex-ocr.txt"
    }
    "layout" = @{
        EnvPath = ".venv-doclayout"
        ReqPath = "deployment/requirements/doclayout.txt"
    }
}

if ($Target -eq "all") {
    $targets = @("dots", "lighton", "paddle-tf", "paddle-paddle", "zhen", "layout")
} else {
    $targets = @($Target)
}

foreach ($name in $targets) {
    $envPath = $specs[$name].EnvPath
    $reqPath = $specs[$name].ReqPath

    if (!(Test-Path $envPath)) {
        Write-Host "==> [$name] creating env $envPath"
        uv venv --seed --python $PythonExe $envPath
    } else {
        Write-Host "==> [$name] reusing env $envPath"
    }

    if (-not $CreateOnly) {
        $pythonPath = Join-Path $root $envPath
        $pythonPath = Join-Path $pythonPath "Scripts/python.exe"
        if (-not $SkipTorch) {
            Write-Host "==> [$name] installing torch/cu121"
            & $pythonPath -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
        }
        if ($name -eq "paddle-paddle") {
            Write-Host "==> [$name] installing paddlepaddle-gpu from official wheel index"
            & $pythonPath -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
        }
        Write-Host "==> [$name] installing common eval deps"
        & $pythonPath -m pip install -r "deployment/requirements/eval-common.txt"
        Write-Host "==> [$name] installing deps from $reqPath"
        & $pythonPath -m pip install -r $reqPath
    }
}

Write-Host "Done."
