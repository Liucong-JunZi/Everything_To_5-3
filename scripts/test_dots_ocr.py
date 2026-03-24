#!/usr/bin/env python
"""
dots.ocr-1.5 测试脚本
完全按照官方 README 示例编写
https://huggingface.co/davanstrien/dots.ocr-1.5
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
from pathlib import Path
import sys

# 模型路径
MODEL_PATH = "data/model/dots.ocr-1.5"
NUM_SAMPLES = 10  # 测试样本数量

def main():
    print("=" * 80)
    print("dots.ocr-1.5 测试脚本")
    print(f"测试样本数: {NUM_SAMPLES}")
    print("=" * 80)

    # 1. 加载模型（README 第 134-140 行）
    print("\n[1/5] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",  # README 要求
        torch_dtype=torch.bfloat16,               # README 要求
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 手动加载 chat_template（如果需要）
    chat_template_path = Path(MODEL_PATH) / "chat_template.json"
    if chat_template_path.exists():
        with open(chat_template_path, 'r', encoding='utf-8') as f:
            processor.chat_template = json.load(f)['chat_template']
        print("✓ 手动加载 chat_template")

    print("✓ 模型加载成功")

    # 2. 准备测试数据
    print(f"\n[2/5] 准备测试数据 ({NUM_SAMPLES}个样本)...")

    # 尝试加载测试图片
    test_images = list(Path("data/HME100K/test_images").glob("*.jpg"))
    if not test_images:
        print("  [错误] 未找到测试图片")
        sys.exit(1)

    test_images = test_images[:NUM_SAMPLES]
    print(f"  找到 {len(test_images)} 个测试图片")

    # 3. 批量处理测试图片
    print("\n[3/5] 开始批量测试...")
    results = []

    for i, img_path in enumerate(test_images, 1):
        print(f"\n  [{i}/{len(test_images)}] 处理: {img_path.name}")

        try:
            # 加载图片
            test_img = Image.open(img_path).convert('RGB')

            # 构建消息（README 第 143-149 行）
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": test_img},
                    {"type": "text", "text": "Extract the text content from this image."},
                ],
            }]

            # 处理输入（README 第 151-153 行）
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            # 生成输出（README 第 155-160 行）
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=24000)  # README 要求 24000

            # 特殊的解码方式（README 第 156-159 行）
            output = processor.batch_decode(
                [generated_ids[0][inputs["input_ids"].shape[-1]:]],
                skip_special_tokens=True,
            )[0]

            results.append({
                "image": img_path.name,
                "output": output[:200] if output else "(空输出)",
                "success": bool(output)
            })

            print(f"    ✓ 成功 - 输出长度: {len(output)}")

        except Exception as e:
            print(f"    ✗ 失败 - {str(e)}")
            results.append({
                "image": img_path.name,
                "output": f"错误: {str(e)}",
                "success": False
            })

    # 4. 输出结果汇总
    print(f"\n[4/5] 测试结果汇总:")
    print("-" * 80)
    success_count = sum(1 for r in results if r["success"])
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {len(results) - success_count}/{len(results)}")

    # 5. 保存详细结果
    print(f"\n[5/5] 保存结果...")
    output_file = Path("data/output") / f"dots_ocr_test_{NUM_SAMPLES}_samples.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "dots.ocr-1.5",
            "samples": NUM_SAMPLES,
            "success_rate": success_count / len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"✓ 结果已保存到: {output_file}")

    print("\n✓ 测试完成")

if __name__ == "__main__":
    main()