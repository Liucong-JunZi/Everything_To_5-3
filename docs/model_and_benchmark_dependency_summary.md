# data/model 与 data/ocrbenchmark 阅读总结

日期: 2026-03-17

## 1) 本次覆盖范围

- `data/model/dots.ocr-1.5`
- `data/model/LightOnOCR-2-1B`
- `data/model/PaddleOCR-VL-1.5`
- `data/model/PP-DocLayoutV3_safetensors`
- `data/model/ZhEn-Latex-OCR`
- `data/ocrbenchmark/ocr-benchmark`
- `data/ocrbenchmark/LaTeX_OCR`
- `data/ocrbenchmark/TC11_package/evaluationTools/lgeval`

## 2) 统一环境建议（主线）

主线目标: 先把当前目录里的主流 OCR/VLM 模型跑通（优先 Transformers + PyTorch 方案）。

建议依赖已写入 `requirements.txt`，核心包括:

- 推理框架: `torch`, `torchvision`, `transformers>=5.0.0`
- 通用库: `numpy`, `pillow`, `requests`, `safetensors`, `sentencepiece`
- 数据/评测辅助: `datasets`, `pyarrow`
- 模型特定:
  - `qwen-vl-utils`（dots.ocr README 示例）
  - `pypdfium2`（LightOnOCR PDF 场景）
  - `flash-attn`（dots.ocr 代码与 PaddleOCR-VL README 都提到）
  - `vllm`（dots/LightOn/PaddleOCR-VL 都支持服务化）
- Paddle 官方链路:
  - `paddlepaddle>=3.2.1`
  - `paddleocr[doc-parser]`

## 3) 模型/基准逐项结论

### 3.1 dots.ocr-1.5

- 代码存在 `from flash_attn import flash_attn_varlen_func`，说明运行时依赖 `flash-attn`。
- README 提供 Transformers 与 vLLM 两种方案，并使用 `qwen_vl_utils`。
- 结论: 必要为 `torch + transformers + qwen-vl-utils`，建议加 `flash-attn` 与 `vllm`。

### 3.2 LightOnOCR-2-1B

- README 明确提到依赖 Transformers v5。
- 示例使用 `pypdfium2` 将 PDF 渲染为图像。
- 结论: `transformers>=5.0.0`、`pypdfium2` 需纳入统一环境。

### 3.3 PaddleOCR-VL-1.5

- README 提供两条路径:
  1. Paddle 官方路径: `paddlepaddle + paddleocr[doc-parser]`
  2. Transformers 路径: `transformers>=5.0.0`
- 另外给了 `flash-attn` 的可选加速方案，也支持 vLLM server。
- 结论: 为避免后续切换成本，统一纳入 Paddle 路径依赖和 Transformers 路径依赖。

### 3.4 PP-DocLayoutV3_safetensors

- 作为 PaddleOCR-VL 的布局模块权重，配合 Transformers/Paddle 生态使用。
- 结论: 无新增独立依赖，复用上面主线环境。

### 3.5 ZhEn-Latex-OCR

- README 示例使用 `VisionEncoderDecoderModel + AutoImageProcessor + AutoTokenizer`。
- 示例中含 `requests` 拉取远程图片。
- 结论: 复用 `transformers + requests + pillow + torch` 即可。

### 3.6 ocr-benchmark / LaTeX_OCR 数据集

- `ocr-benchmark` 当前目录主要是图片+标注文件。
- `LaTeX_OCR` 是 parquet 数据格式。
- 结论: 建议统一带上 `datasets + pyarrow`，读取和评测更方便。

### 3.7 TC11 lgeval 老评测工具链

- README 明确依赖: `python 2.6/2.7`, `bash`, `perl + LibXML`, `Graphviz`, `TXL`。
- 这是老链路，不建议与主环境混装。
- 结论: 建议后续单独建 legacy 环境（甚至容器）来跑 TC11 工具。

## 4) 实施建议

- 先安装 `requirements.txt` 跑通主线模型对比。
- 若要使用 TC11 的老评测脚本，再单独准备 Python2 + 系统工具环境。
- `flash-attn` 与 `vllm` 在 Windows/特定 CUDA 上可能有额外安装门槛，必要时可先跳过，先跑基础 Transformers 推理。
