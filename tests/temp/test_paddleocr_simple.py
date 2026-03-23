"""
PaddleOCR-VL-1.5 简单测试脚本
使用标准 transformers API
"""
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

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
        trust_remote_code=False  # 不使用自定义代码
    )

    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=False,  # 不使用自定义代码
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded on {device}")
    return processor, model, device


def predict(processor, model, device, image, prompt):
    """使用模型进行预测"""
    # 简单的输入处理
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    # 移动到设备
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    # 解码
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
                try:
                    pred = predict(processor, model, device, image, prompt)
                    gt = row["true_markdown_output"]

                    results.append({
                        "sample": image_rel,
                        "task": "markdown",
                        "prompt": prompt,
                        "prediction": pred[:200],
                        "ground_truth": gt[:200],
                        "pred_len": len(pred),
                        "gt_len": len(gt),
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "sample": image_rel,
                        "task": "markdown",
                        "status": "error",
                        "error": str(e)
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
            try:
                pred = predict(processor, model, device, image, prompt)

                results.append({
                    "sample": name,
                    "task": "formula",
                    "prompt": prompt,
                    "prediction": pred,
                    "ground_truth": label,
                    "pred_len": len(pred),
                    "gt_len": len(label),
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "sample": name,
                    "task": "formula",
                    "status": "error",
                    "error": str(e)
                })
            count += 1

    return results


def main():
    print("=" * 60)
    print("PaddleOCR-VL-1.5 测试脚本")
    print("=" * 60)

    try:
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
            print(f"Status: {r['status']}")
            if r['status'] == 'success':
                print(f"Task: {r['task']}")
                print(f"Prompt: {r['prompt']}")
                print(f"Prediction ({r['pred_len']} chars): {r['prediction']}")
                print(f"Ground Truth ({r['gt_len']} chars): {r['ground_truth']}")
            else:
                print(f"Error: {r.get('error', 'Unknown')}")

        print("\n[HME100K Results]")
        for r in hme_results:
            print(f"\nSample: {r['sample']}")
            print(f"Status: {r['status']}")
            if r['status'] == 'success':
                print(f"Task: {r['task']}")
                print(f"Prompt: {r['prompt']}")
                print(f"Prediction: {r['prediction']}")
                print(f"Ground Truth: {r['ground_truth']}")
            else:
                print(f"Error: {r.get('error', 'Unknown')}")

        # 保存结果
        output_path = SCRIPT_ROOT / "test_paddleocr_results.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "ocr_benchmark": ocr_results,
                "hme100k": hme_results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] Results saved to {output_path}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
