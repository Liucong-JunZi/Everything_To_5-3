"""
dots.ocr-1.5 单独测试脚本
诊断为什么预测结果为空
"""
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# 设置环境
SCRIPT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(SCRIPT_ROOT / "data" / ".hf_cache"))

MODEL_PATH = SCRIPT_ROOT / "data" / "model" / "dots.ocr-1.5"
DATA_ROOT = SCRIPT_ROOT / "data" / "ocrbenchmark"


def test_single_image():
    """测试单张图片"""
    print("=" * 60)
    print("dots.ocr-1.5 单图测试")
    print("=" * 60)

    # 加载模型
    print(f"\n[INFO] Loading model from {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded on {device}")

    # 测试图片
    test_image_path = DATA_ROOT / "HME100K" / "test.zip"

    # 从 HME100K 提取一张图片
    import zipfile
    from io import BytesIO

    with zipfile.ZipFile(test_image_path) as zf:
        # 读取第一张图片
        with zf.open("test_images/test_2.jpg") as fp:
            image = Image.open(BytesIO(fp.read())).convert("RGB")

    print(f"\n[INFO] Image size: {image.size}")

    # 测试不同的 prompt
    prompts = [
        "Recognize the mathematical expression and output LaTeX only.",
        "What is in this image?",
        "OCR:",
        "",  # 空 prompt
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[TEST {i}] Prompt: '{prompt}'")

        try:
            # 构建消息
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

            print(f"  Template output length: {len(text)}")

            # 处理输入
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )

            print(f"  Input IDs shape: {inputs['input_ids'].shape}")
            print(f"  Input IDs length: {inputs['input_ids'].shape[-1]}")

            # 移动到设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            # 过滤不支持的参数
            generate_inputs = {}
            supported_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
            for k, v in inputs.items():
                if k in supported_keys:
                    generate_inputs[k] = v
                else:
                    print(f"  Filtered out: {k}")

            # 生成
            input_length = generate_inputs["input_ids"].shape[-1]
            print(f"  Generating with max_new_tokens=100...")

            with torch.no_grad():
                output_ids = model.generate(
                    **generate_inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=None,  # 禁用 EOS，强制生成
                    pad_token_id=processor.tokenizer.pad_token_id
                )

            print(f"  Output IDs shape: {output_ids.shape}")
            print(f"  Output length: {output_ids.shape[-1]}")
            print(f"  New tokens: {output_ids.shape[-1] - input_length}")

            # 解码（去掉输入部分）
            output_ids = output_ids[:, input_length:]
            text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            print(f"  Result length: {len(text)}")
            print(f"  Result: '{text}'")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_single_image()
