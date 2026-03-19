"""
PaddleOCR-VL-1.5 单独测试脚本
测试模型在不同数据集上的表现
"""
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# 设置环境
SCRIPT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(SCRIPT_ROOT / "data" / ".hf_cache"))

MODEL_PATH = SCRIPT_ROOT / "data" / "model" / "PaddleOCR-VL-1.5"
DATA_ROOT = SCRIPT_ROOT / "data" / "ocrbenchmark"


def load_model():
    """加载 PaddleOCR-VL-1.5 模型"""
    print(f"[INFO] Loading PaddleOCR-VL-1.5 from {MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa"
    )
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded on {device}")
    return processor, model, device


def predict(processor, model, device, image, prompt):
    """使用模型进行预测"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 应用 chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 处理输入
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # 移动到设备
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # 过滤不支持的参数
    generate_inputs = {}
    supported_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    for k, v in inputs.items():
        if k in supported_keys:
            generate_inputs[k] = v

    # 生成
    input_length = generate_inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(
            **generate_inputs,
            max_new_tokens=512,
            do_sample=False
        )

    # 解码（去掉输入部分）
    output_ids = output_ids[:, input_length:]
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return text.strip()


def test_ocr_benchmark(processor, model, device, num_samples=2):
    """测试 ocr-benchmark 数据集"""
    print(f"\n[TEST] OCR Benchmark - {num_samples} samples")

    meta_path = DATA_ROOT / "ocr-benchmark" / "test" / "metadata.jsonl"
    image_root = DATA_ROOT / "ocr-benchmark" / "test"

    results = []
    count = 0

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if count >= num_samples:
                break

            row = json.loads(line)
            image_rel = row.get("file_name")
            if not image_rel:
                continue

            image_path = image_root / image_rel
            if not image_path.exists():
                continue

            image = Image.open(image_path).convert("RGB")

            # 测试 markdown 提取
            if row.get("true_markdown_output"):
                prompt = "OCR:"
                print(f"  Sample {count+1}: {image_rel} (markdown)")
                pred = predict(processor, model, device, image, prompt)
                gt = row["true_markdown_output"]

                results.append({
                    "sample": image_rel,
                    "task": "markdown",
                    "prompt": prompt,
                    "prediction": pred[:200],
                    "ground_truth": gt[:200],
                    "pred_len": len(pred),
                    "gt_len": len(gt)
                })
                count += 1

                if count >= num_samples:
                    break

    return results


def test_hme100k(processor, model, device, num_samples=2):
    """测试 HME100K 数据集（手写数学公式）"""
    print(f"\n[TEST] HME100K - {num_samples} samples")

    zip_path = DATA_ROOT / "HME100K" / "test.zip"

    results = []
    count = 0

    with zipfile.ZipFile(zip_path) as zf:
        lines = zf.read("test_labels.txt").decode("utf-8", errors="replace").splitlines()

        for line in lines:
            if count >= num_samples:
                break

            if "\t" not in line:
                continue

            name, label = line.split("\t", 1)
            archive_name = f"test_images/{name}"

            # 从 zip 读取图片
            with zf.open(archive_name) as fp:
                image = Image.open(BytesIO(fp.read())).convert("RGB")

            prompt = "Formula Recognition:"
            print(f"  Sample {count+1}: {name}")
            pred = predict(processor, model, device, image, prompt)

            results.append({
                "sample": name,
                "task": "formula",
                "prompt": prompt,
                "prediction": pred,
                "ground_truth": label,
                "pred_len": len(pred),
                "gt_len": len(label)
            })
            count += 1

    return results


def main():
    print("=" * 60)
    print("PaddleOCR-VL-1.5 测试脚本")
    print("=" * 60)

    # 加载模型
    processor, model, device = load_model()

    # 测试 OCR Benchmark
    ocr_results = test_ocr_benchmark(processor, model, device, num_samples=2)

    # 测试 HME100K
    hme_results = test_hme100k(processor, model, device, num_samples=2)

    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    print("\n[OCR Benchmark Results]")
    for r in ocr_results:
        print(f"\nSample: {r['sample']}")
        print(f"Task: {r['task']}")
        print(f"Prompt: {r['prompt']}")
        print(f"Prediction ({r['pred_len']} chars): {r['prediction']}")
        print(f"Ground Truth ({r['gt_len']} chars): {r['ground_truth']}")

    print("\n[HME100K Results]")
    for r in hme_results:
        print(f"\nSample: {r['sample']}")
        print(f"Task: {r['task']}")
        print(f"Prompt: {r['prompt']}")
        print(f"Prediction: {r['prediction']}")
        print(f"Ground Truth: {r['ground_truth']}")

    # 保存结果
    output_path = SCRIPT_ROOT / "test_paddleocr_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "ocr_benchmark": ocr_results,
            "hme100k": hme_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
