param(
    [string]$Datasets = "ocr-benchmark,HME100K,LaTeX_OCR",
    [string]$ModelRoot = "data/model",
    [string]$OutputDir = "data/output",
    [string]$DocsDir = "docs"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$modelToPy = @{
    "dots.ocr-1.5" = ".venv-dots-ocr\Scripts\python.exe"
    "LightOnOCR-2-1B" = ".venv-lighton-ocr\Scripts\python.exe"
    "PaddleOCR-VL-1.5" = ".venv-paddleocr-vl-tf\Scripts\python.exe"
    "PP-DocLayoutV3_safetensors" = ".venv-doclayout\Scripts\python.exe"
    "ZhEn-Latex-OCR" = ".venv-zhen-latex-ocr\Scripts\python.exe"
}

$models = @()
Get-ChildItem $ModelRoot -Directory | ForEach-Object {
    if ($modelToPy.ContainsKey($_.Name)) {
        $models += $_.Name
    }
}

if ($models.Count -eq 0) {
    throw "No supported model folders found under $ModelRoot"
}

$combinedDir = Join-Path $OutputDir "per_model"
New-Item -ItemType Directory -Force -Path $combinedDir | Out-Null

foreach ($m in $models) {
    $py = $modelToPy[$m]
    if (!(Test-Path $py)) {
        Write-Warning "Skip ${m}: python not found at $py"
        continue
    }
    $out = Join-Path $combinedDir $m
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Write-Host "==> running full benchmark for $m"
    & $py scripts/eval_matrix.py `
        --mode full `
        --models $m `
        --datasets $Datasets `
        --model-root $ModelRoot `
        --output-dir $out `
        --docs-dir $DocsDir `
        --max-samples-per-dataset 500
}

$mergeScript = @'
import json
from pathlib import Path
import pandas as pd

base = Path("data/output/per_model")
rows = []
detail = {}
for model_dir in base.iterdir():
    if not model_dir.is_dir():
        continue
    csv_path = model_dir / "eval_results.csv"
    js_path = model_dir / "eval_results.json"
    if csv_path.exists():
        rows.append(pd.read_csv(csv_path))
    if js_path.exists():
        payload = json.loads(js_path.read_text(encoding="utf-8"))
        detail.update(payload.get("details", {}))

if not rows:
    raise SystemExit("No per-model results found")

summary = pd.concat(rows, ignore_index=True)
summary = summary.sort_values(["dataset", "task", "model"]).reset_index(drop=True)
summary.to_csv("data/output/eval_results.csv", index=False, encoding="utf-8")

Path("data/output/eval_results.json").write_text(
    json.dumps({"summary": summary.to_dict(orient="records"), "details": detail}, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("merged:", len(summary))
'@

& .venv-lighton-ocr\Scripts\python.exe -c $mergeScript

$reportScript = @'
from pathlib import Path
import pandas as pd
from scripts.eval_matrix import plot_outputs, write_report

output_dir = Path("data/output")
docs_dir = Path("docs")
figures_dir = docs_dir / "figures"
summary = pd.read_csv(output_dir / "eval_results.csv")
figs = plot_outputs(summary, figures_dir)
write_report(docs_dir / "benchmark_report.md", "full", summary, figs, output_dir / "eval_results.csv", output_dir / "eval_results.json")
print("report updated")
'@

& .venv-lighton-ocr\Scripts\python.exe -c $reportScript

Write-Host "Done."
