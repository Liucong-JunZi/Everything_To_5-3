# OCR 模型独立测试脚本

## 概述

每个模型都有独立的测试脚本，完全按照官方 README 示例编写，避免复杂的适配器逻辑。

## 脚本列表

| 模型 | 脚本 | 虚拟环境 | 官方文档 |
|------|------|----------|----------|
| dots.ocr-1.5 | `scripts/test_dots_ocr.py` | `.venv-dots-ocr` | [README](data/model/dots.ocr-1.5/README.md) |
| LightOnOCR-2-1B | `scripts/test_lighton_ocr.py` | `.venv-lighton-ocr` | [README](data/model/LightOnOCR-2-1B/README.md) |
| ZhEn-Latex-OCR | `scripts/test_zhen_latex.py` | `.venv-zhen-latex-ocr` | [README](data/model/ZhEn-Latex-OCR/README.md) |
| PaddleOCR-VL-1.5 | `scripts/test_paddle_vl.py` | `.venv-paddleocr-vl-tf` | [README](data/model/PaddleOCR-VL-1.5/README.md) |

## 快速开始

### 方法1：批量运行（Windows）
```batch
run_all_tests.bat
```

### 方法2：单独运行
```bash
# dots.ocr-1.5
.venv-dots-ocr\Scripts\python.exe scripts\test_dots_ocr.py

# LightOnOCR-2-1B
.venv-lighton-ocr\Scripts\python.exe scripts\test_lighton_ocr.py

# ZhEn-Latex-OCR
.venv-zhen-latex-ocr\Scripts\python.exe scripts\test_zhen_latex.py

# PaddleOCR-VL-1.5
.venv-paddleocr-vl-tf\Scripts\python.exe scripts\test_paddle_vl.py
```

## 脚本特点

1. **完全基于官方 README**：每个脚本都严格按照官方文档的示例代码编写
2. **独立运行**：不依赖复杂的适配器系统
3. **易于调试**：每个脚本都是独立的，出问题容易定位
4. **环境隔离**：每个模型使用独立的虚拟环境，避免依赖冲突

## 测试数据

脚本会自动查找测试数据：
- 优先使用 `data/HME100K/test_images/` 中的图片
- 如果找不到，会创建空白测试图片

## 自定义测试

### 修改测试图片
编辑脚本中的 `TEST_IMAGE` 变量，或者直接修改测试图片路径。

### 修改生成参数
每个脚本中都有 `max_new_tokens` 参数，可以根据需要调整：
- dots.ocr-1.5: 24000 (README 要求)
- LightOnOCR-2-1B: 512
- ZhEn-Latex-OCR: 128 (README 建议 100-300)
- PaddleOCR-VL-1.5: 512

## 环境配置

如果虚拟环境不存在，可以使用以下命令创建：

```bash
# dots.ocr-1.5
python -m venv .venv-dots-ocr
.venv-dots-ocr\Scripts\activate
pip install torch transformers qwen-vl-utils

# LightOnOCR-2-1B
python -m venv .venv-lighton-ocr
.venv-lighton-ocr\Scripts\activate
pip install torch transformers

# ZhEn-Latex-OCR
python -m venv .venv-zhen-latex-ocr
.venv-zhen-latex-ocr\Scripts\activate
pip install torch transformers

# PaddleOCR-VL-1.5
python -m venv .venv-paddleocr-vl-tf
.venv-paddleocr-vl-tf\Scripts\activate
pip install torch transformers torchvision
```

## 常见问题

### Q: 为什么每个模型需要独立的虚拟环境？
A: 不同模型可能依赖不同版本的库，独立环境可以避免版本冲突。

### Q: 脚本运行失败怎么办？
A:
1. 检查虚拟环境是否存在
2. 检查依赖是否安装完整
3. 检查模型文件是否完整
4. 查看错误信息，通常会有明确的提示

### Q: 如何集成到评估系统？
A: 每个脚本都可以独立运行，也可以作为模块导入。评估系统可以调用这些脚本获取预测结果。

## 对比旧的 eval_matrix.py

| 特性 | 旧系统 (eval_matrix.py) | 新系统 (独立脚本) |
|------|------------------------|------------------|
| 复杂度 | 高（所有模型在一个文件） | 低（每个模型独立） |
| 调试难度 | 高 | 低 |
| 代码维护 | 困难 | 容易 |
| 遵循官方示例 | 部分遵循 | 完全遵循 |
| 环境隔离 | 无 | 完全隔离 |