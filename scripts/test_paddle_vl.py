#!/usr/bin/env python
"""
PaddleOCR-VL-1.5 测试脚本 (transformers 版本)
完全按照官方 README 示例编写
https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from pathlib import Path
import json
import sys

# 模型路径
MODEL_PATH = "data/model/PaddleOCR-VL-1.5"
NUM_SAMPLES = 10  # 测试样本数量

def main():
    print("=" * 80)
    print("PaddleOCR-VL-1.5 测试脚本 (transformers)")
    print(f"测试样本数: {NUM_SAMPLES}")
    print("=" * 80)

    # 1. 加载模型（README 第 206-207 行）
    print("\n[1/4] 加载模型...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"✓ 模型加载成功 (device: {DEVICE})")

    # 2. 准备测试数据
    print(f"\n[2/4] 准备测试数据 ({NUM_SAMPLES}个样本)...")
    test_images = list(Path("data/HME100K/test_images").glob("*.jpg"))
    if not test_images:
        print("  [错误] 未找到测试图片")
        sys.exit(1)

    test_images = test_images[:NUM_SAMPLES]
    print(f"  找到 {len(test_images)} 个测试图片")

    # 3. 批量测试
    print("\n[3/4] 开始批量测试...")
    results = []

    task = "ocr"  # README 第 175 行：可选 'ocr' | 'table' | 'chart' | 'formula' | 'spotting' | 'seal'
    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "spotting": "Spotting:",
        "seal": "Seal Recognition:",
    }

    for i, img_path in enumerate(test_images, 1):
        print(f"\n  [{i}/{len(test_images)}] 处理: {img_path.name}")

        try:
            test_img = Image.open(img_path).convert('RGB')

            # 处理输入（README 第 209-225 行）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_img},
                        {"type": "text", "text": PROMPTS[task]},
                    ]
                }
            ]

            # README 第 218-225 行
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": processor.image_processor.min_pixels,
                        "longest_edge": 1280 * 28 * 28
                    }
                },
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512)

            # README 第 228 行：去掉最后一个 token
            result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1], skip_special_tokens=True)

            results.append({
                "image": img_path.name,
                "output": result[:200] if result else "(空输出)",
                "success": bool(result)
            })

            print(f"    ✓ 成功 - 输出长度: {len(result)}")

        except Exception as e:
            print(f"    ✗ 失败 - {str(e)}")
            results.append({
                "image": img_path.name,
                "output": f"错误: {str(e)}",
                "success": False
            })

    # 4. 输出结果汇总
    print(f"\n[4/4] 测试结果汇总:")
    print("-" * 80)
    success_count = sum(1 for r in results if r["success"])
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {len(results) - success_count}/{len(results)}")

    # 保存结果
    output_file = Path("data/output") / f"paddle_vl_test_{NUM_SAMPLES}_samples.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "PaddleOCR-VL-1.5",
            "samples": NUM_SAMPLES,
            "success_rate": success_count / len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {output_file}")

    print("\n✓ 测试完成")

if __name__ == "__main__":
    main()