# Per-model uv environments

This project now supports separate uv virtual environments per model in project root:

- `.venv-dots-ocr`
- `.venv-lighton-ocr`
- `.venv-paddleocr-vl-tf`
- `.venv-paddleocr-vl-paddle`
- `.venv-zhen-latex-ocr`
- `.venv-doclayout`

## Create all envs

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_model_envs.ps1
```

## Create envs only (no package install)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_model_envs.ps1 -CreateOnly
```

## Create one env

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_model_envs.ps1 -Target dots
```

Available targets:

- `dots`
- `lighton`
- `paddle-tf`
- `paddle-paddle`
- `zhen`
- `layout`
