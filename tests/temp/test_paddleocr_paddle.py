"""
PaddleOCR-VL-1.5 测试脚本 - 使用 PaddleOCR 框架
"""
import json
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from paddleocr import PaddleOCR

SCRIPT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_ROOT / "data" / "model" / "PaddleOCR-VL-1.5"
DATA_ROOT = SCRIPT_ROOT / "data" / "ocrbenchmark"


def test_ocr_benchmark(ocr, num_samples=2):
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

            # 测试 markdown 提取
            if row.get("true_markdown_output"):
                print(f"  Sample {count+1}: {image_rel} (markdown)")
                try:
                    result = ocr.predict(str(image_path))

                    # 提取文本
                    pred_text = ""
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                pred_text += line[1][0] + "\n"

                    gt = row["true_markdown_output"]

                    results.append({
                        "sample": image_rel,
                        "task": "markdown",
                        "prediction": pred_text[:200],
                        "ground_truth": gt[:200],
                        "pred_len": len(pred_text),
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


def test_hme100k(ocr, num_samples=2):
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

            print(f"  Sample {count+1}: {name}")
            try:
                # 从 zip 读取图片
                with zf.open(archive_name) as fp:
                    image_bytes = fp.read()
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")

                    # 保存临时文件
                    temp_path = SCRIPT_ROOT / "temp_image.jpg"
                    image.save(temp_path)

                    # OCR 识别
                    result = ocr.predict(str(temp_path))

                    # 提取文本
                    pred_text = ""
                    if result and result[0]:
                        for line_result in result[0]:
                            if line_result and len(line_result) >= 2:
                                pred_text += line_result[1][0] + " "

                    # 删除临时文件
                    temp_path.unlink()

                results.append({
                    "sample": name,
                    "task": "formula",
                    "prediction": pred_text.strip(),
                    "ground_truth": label,
                    "pred_len": len(pred_text),
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
    print("PaddleOCR-VL-1.5 测试脚本 (PaddleOCR 框架)")
    print("=" * 60)

    try:
        # 初始化 PaddleOCR
        print(f"\n[INFO] Initializing PaddleOCR...")
        ocr = PaddleOCR(lang='ch', device='gpu')
        print(f"[INFO] PaddleOCR initialized")

        # 测试 OCR Benchmark
        ocr_results = test_ocr_benchmark(ocr, num_samples=2)

        # 测试 HME100K
        hme_results = test_hme100k(ocr, num_samples=2)

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
                print(f"Prediction: {r['prediction']}")
                print(f"Ground Truth: {r['ground_truth']}")
            else:
                print(f"Error: {r.get('error', 'Unknown')}")

        # 保存结果
        output_path = SCRIPT_ROOT / "test_paddleocr_paddle_results.json"
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
