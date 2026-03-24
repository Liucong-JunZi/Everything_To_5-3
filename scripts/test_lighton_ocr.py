#!/usr/bin/env python
"""
LightOnOCR-2-1B 测试脚本
完全按照官方 README 示例编写
https://huggingface.co/lightonai/LightOnOCR-2-1B
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from pathlib import Path
import json
import sys

# 模型路径
MODEL_PATH = "data/model/LightOnOCR-2-1B"
NUM_SAMPLES = 10  # 测试样本数量

def main():
    print("=" * 80)
    print("LightOnOCR-2-1B 测试脚本")
    print(f"测试样本数: {NUM_SAMPLES}")
    print("=" * 80)

    # 1. 加载模型（README 示例）
    print("\n[1/4] 加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"✓ 模型加载成功 (device: {device}, dtype: {dtype})")

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

            # 处理输入（README 示例）
            inputs = processor(images=test_img, text="OCR:", return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                      for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512)

            result = processor.decode(outputs[0], skip_special_tokens=True)

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
    output_file = Path("data/output") / f"lighton_ocr_test_{NUM_SAMPLES}_samples.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "LightOnOCR-2-1B",
            "samples": NUM_SAMPLES,
            "success_rate": success_count / len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {output_file}")

    print("\n✓ 测试完成")

if __name__ == "__main__":
    main()