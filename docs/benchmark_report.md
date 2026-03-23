# Benchmark Report

- generated_at: 2026-03-23T17:03:43
- mode: smoke
- summary_csv: `data/output/test_dots_clean/eval_results.csv`
- summary_json: `data/output/test_dots_clean/eval_results.json`

## Task-level scores

```csv
model,dataset,task,mean_score,samples,success_count,error_count,na_count,json_parse_rate,note
dots.ocr-1.5,HME100K,formula_ocr,0.0000,2,2,0,0,,
dots.ocr-1.5,LaTeX_OCR,formula_ocr,0.0000,2,2,0,0,,
dots.ocr-1.5,TC11_package,legacy_struct_eval,,0,0,0,0,,pending_legacy_eval
dots.ocr-1.5,ocr-benchmark,json_extraction,0.0000,1,0,1,0,,
dots.ocr-1.5,ocr-benchmark,markdown_extraction,0.0000,1,0,1,0,,

```

## Figures

![score_heatmap.png](docs/figures/score_heatmap.png)

![task_bars_formula_ocr.png](docs/figures/task_bars_formula_ocr.png)

![task_bars_json_extraction.png](docs/figures/task_bars_json_extraction.png)

![task_bars_markdown_extraction.png](docs/figures/task_bars_markdown_extraction.png)
