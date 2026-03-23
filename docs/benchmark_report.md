# Benchmark Report

- generated_at: 2026-03-23T20:07:28
- mode: smoke
- summary_csv: `data/output/test_zhen_fixed/eval_results.csv`
- summary_json: `data/output/test_zhen_fixed/eval_results.json`

## Task-level scores

```csv
model,dataset,task,mean_score,samples,success_count,error_count,na_count,json_parse_rate,note
ZhEn-Latex-OCR,HME100K,formula_ocr,0.0000,3,0,3,0,,
ZhEn-Latex-OCR,LaTeX_OCR,formula_ocr,0.0000,3,0,3,0,,
ZhEn-Latex-OCR,TC11_package,legacy_struct_eval,,0,0,0,0,,pending_legacy_eval
ZhEn-Latex-OCR,ocr-benchmark,json_extraction,,1,0,0,1,,not_supported
ZhEn-Latex-OCR,ocr-benchmark,markdown_extraction,,2,0,0,2,,not_supported

```

## Figures

![score_heatmap.png](docs/figures/score_heatmap.png)

![task_bars_formula_ocr.png](docs/figures/task_bars_formula_ocr.png)
