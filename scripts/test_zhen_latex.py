#!/usr/bin/env python
"""
ZhEn-Latex-OCR 测试脚本
完全按照官方 README 示例编写
https://huggingface.co/MixTex/ZhEn-Latex-OCR
"""

from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor
from PIL import Image
import torch
from pathlib import Path
import json
import sys

# 模型路径
MODEL_PATH = "data/model/ZhEn-Latex-OCR"
NUM_SAMPLES = 10  # 测试样本数量

def main():
    print("=" * 80)
    print("ZhEn-Latex-OCR 测试脚本")
    print(f"测试样本数: {NUM_SAMPLES}")
    print("=" * 80)

    # 1. 加载模型（README 第 19-21 行）
    print("\n[1/4] 加载模型...")
    feature_extractor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, max_len=296)  # README 要求
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

    print("✓ 模型加载成功")

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

    for i, img_path in enumerate(test_images, 1):
        print(f"\n  [{i}/{len(test_images)}] 处理: {img_path.name}")

        try:
            test_img = Image.open(img_path).convert('RGB')

            # README 第 25 行
            with torch.no_grad():
                output = model.generate(
                    feature_extractor(test_img, return_tensors="pt").pixel_values,
                    max_new_tokens=128
                )

            # README 第 25 行的后处理
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            result = result.replace('\\[', '\\begin{align*}').replace('\\]', '\\end{align*}')

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
    output_file = Path("data/output") / f"zhen_latex_test_{NUM_SAMPLES}_samples.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "ZhEn-Latex-OCR",
            "samples": NUM_SAMPLES,
            "success_rate": success_count / len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {output_file}")

    print("\n✓ 测试完成")

if __name__ == "__main__":
    main()