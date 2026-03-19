# Quick Reference - Benchmark Evaluation

## Current Status
✅ Benchmark is running in background
✅ Limited to 500 samples per dataset
✅ All 5 models will be evaluated

## Models Being Evaluated
1. **dots.ocr-1.5** - Currently running
2. **LightOnOCR-2-1B** - Queued
3. **PaddleOCR-VL-1.5** - Queued
4. **PP-DocLayoutV3_safetensors** - Queued
5. **ZhEn-Latex-OCR** - Queued

## Datasets (500 samples each)
- `ocr-benchmark` - Document OCR (markdown/JSON)
- `HME100K` - Handwritten math expressions
- `LaTeX_OCR` - LaTeX formulas

## Check Progress

### View live output
```bash
tail -f C:\Users\Lenovo\AppData\Local\Temp\claude\F--everyting-to-5-3-Everything-To-5-3\47c7b697-210b-4e3c-9c11-b9f90a345873\tasks\bzskelun3.output
```

### Check completed models
```bash
ls data/output/per_model/
```

### View results as they complete
```bash
cat data/output/per_model/dots.ocr-1.5/eval_results.csv
cat data/output/per_model/LightOnOCR-2-1B/eval_results.csv
# etc.
```

## Final Results Location
After all models complete:
- **Summary CSV**: `data/output/eval_results.csv`
- **Detailed JSON**: `data/output/eval_results.json`
- **Report**: `docs/benchmark_report.md`
- **Figures**: `docs/figures/`

## Manual Run (if needed)

### Run all models
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_benchmark_uv.ps1
```

### Run single model
```powershell
.venv-dots-ocr\Scripts\python.exe scripts/eval_matrix.py --mode full --models "dots.ocr-1.5" --max-samples-per-dataset 500
```

### Run smoke test (20 samples)
```bash
python scripts/eval_matrix.py --mode smoke
```

## Estimated Time
- Per model: ~10-30 minutes (depends on GPU/CPU)
- Total for 5 models: ~1-2.5 hours

## Key Changes Made
1. ✅ Modified `eval_matrix.py` to support `--max-samples-per-dataset` parameter
2. ✅ Updated `run_full_benchmark_uv.ps1` to use 500 samples per dataset
3. ✅ Started benchmark execution in background
