@echo off
REM OCR 模型测试脚本运行器
REM 每个模型使用对应的虚拟环境

echo ================================================================================
echo OCR 模型独立测试脚本
echo ================================================================================
echo.

REM dots.ocr-1.5 测试
echo [1/4] 测试 dots.ocr-1.5
echo 使用环境: .venv-dots-ocr
echo --------------------------------------------------------------------------------
if exist ".venv-dots-ocr\Scripts\python.exe" (
    .venv-dots-ocr\Scripts\python.exe scripts\test_dots_ocr.py
) else (
    echo [错误] 虚拟环境 .venv-dots-ocr 不存在
)
echo.
echo.

REM LightOnOCR-2-1B 测试
echo [2/4] 测试 LightOnOCR-2-1B
echo 使用环境: .venv-lighton-ocr
echo --------------------------------------------------------------------------------
if exist ".venv-lighton-ocr\Scripts\python.exe" (
    .venv-lighton-ocr\Scripts\python.exe scripts\test_lighton_ocr.py
) else (
    echo [错误] 虚拟环境 .venv-lighton-ocr 不存在
)
echo.
echo.

REM ZhEn-Latex-OCR 测试
echo [3/4] 测试 ZhEn-Latex-OCR
echo 使用环境: .venv-zhen-latex-ocr
echo --------------------------------------------------------------------------------
if exist ".venv-zhen-latex-ocr\Scripts\python.exe" (
    .venv-zhen-latex-ocr\Scripts\python.exe scripts\test_zhen_latex.py
) else (
    echo [错误] 虚拟环境 .venv-zhen-latex-ocr 不存在
)
echo.
echo.

REM PaddleOCR-VL-1.5 测试
echo [4/4] 测试 PaddleOCR-VL-1.5 (transformers)
echo 使用环境: .venv-paddleocr-vl-tf
echo --------------------------------------------------------------------------------
if exist ".venv-paddleocr-vl-tf\Scripts\python.exe" (
    .venv-paddleocr-vl-tf\Scripts\python.exe scripts\test_paddle_vl.py
) else (
    echo [错误] 虚拟环境 .venv-paddleocr-vl-tf 不存在
)
echo.

echo ================================================================================
echo 所有测试完成
echo ================================================================================
pause