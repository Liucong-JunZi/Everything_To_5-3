# Docs

- `model_and_benchmark_dependency_summary.md`: `data/model` and `data/ocrbenchmark` dependency reading summary and environment advice.
- `benchmark_report.md`: auto-generated evaluation report with per-task scores and figures.
- `model_envs.md`: per-model `uv` environment setup in project root.

## Evaluation

Run smoke test:

```bash
python scripts/eval_matrix.py --mode smoke
```

Run full evaluation:

```bash
python scripts/eval_matrix.py --mode full
```

Generated outputs:

- `data/output/eval_results.csv`
- `data/output/eval_results.json`
- `docs/benchmark_report.md`
- `docs/figures/score_heatmap.png`
- `docs/figures/task_bars_*.png`
