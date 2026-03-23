# Benchmark Run Summary

## Date: 2026-03-18

## Changes Made

### 1. Modified `scripts/eval_matrix.py`
- Updated `select_samples()` function to support limiting samples per dataset
- Added `--max-samples-per-dataset` argument (default: 0 = unlimited)
- When set to a positive value in full mode, limits each dataset to that many samples

### 2. Modified `scripts/run_full_benchmark_uv.ps1`
- Added `--max-samples-per-dataset 500` parameter to all model evaluation runs
- This ensures each benchmark dataset is limited to 500 samples

## Project Structure

### Models Available (5 models)
1. `dots.ocr-1.5` - Uses `.venv-dots-ocr`
2. `LightOnOCR-2-1B` - Uses `.venv-lighton-ocr`
3. `PaddleOCR-VL-1.5` - Uses `.venv-paddleocr-vl-tf`
4. `PP-DocLayoutV3_safetensors` - Uses `.venv-doclayout`
5. `ZhEn-Latex-OCR` - Uses `.venv-zhen-latex-ocr`

### Datasets Available (4 datasets)
1. `ocr-benchmark` - Document OCR with markdown/JSON extraction
2. `HME100K` - Handwritten mathematical expression recognition
3. `LaTeX_OCR` - LaTeX formula recognition
4. `TC11_package` - Legacy structural evaluation (placeholder)

### Virtual Environments
All model-specific environments are already set up:
- `.venv-dots-ocr`
- `.venv-lighton-ocr`
- `.venv-paddleocr-vl-tf`
- `.venv-paddleocr-vl-paddle`
- `.venv-zhen-latex-ocr`
- `.venv-doclayout`

## Benchmark Execution

### Command
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_benchmark_uv.ps1
```

### Process
1. For each model, runs evaluation using its dedicated Python environment
2. Evaluates against all datasets: `ocr-benchmark`, `HME100K`, `LaTeX_OCR`
3. Limits each dataset to **500 samples** (instead of full dataset)
4. Saves per-model results to `data/output/per_model/{model_name}/`
5. Merges all results into `data/output/eval_results.csv` and `eval_results.json`
6. Generates visualizations in `docs/figures/`
7. Creates final report at `docs/benchmark_report.md`

### Output Files
- `data/output/eval_results.csv` - Summary scores table
- `data/output/eval_results.json` - Detailed results with predictions
- `data/output/per_model/{model}/` - Individual model results
- `docs/benchmark_report.md` - Auto-generated report
- `docs/figures/score_heatmap.png` - Model x Task heatmap
- `docs/figures/task_bars_*.png` - Per-task bar charts

## Sample Limitation Details

With `--max-samples-per-dataset 500`:
- Each dataset is limited to first 500 samples
- Applies independently per dataset (not per task)
- Reduces evaluation time significantly
- Still provides representative performance metrics

## Monitoring Progress

Check background task status:
```bash
# View task output
cat C:\Users\Lenovo\AppData\Local\Temp\claude\F--everyting-to-5-3-Everything-To-5-3\47c7b697-210b-4e3c-9c11-b9f90a345873\tasks\bzskelun3.output
```

Or check per-model output directories as they complete:
```bash
ls data/output/per_model/
```
