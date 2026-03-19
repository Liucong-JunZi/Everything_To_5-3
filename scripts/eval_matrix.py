import argparse
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_HOME = SCRIPT_ROOT / "data" / ".hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_HF_HOME / "modules"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HOME / "hub"))

from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)


DATASET_OCR_BENCH = "ocr-benchmark"
DATASET_HME100K = "HME100K"
DATASET_LATEX_OCR = "LaTeX_OCR"
DATASET_TC11 = "TC11_package"
TASK_TC11_LEGACY = "legacy_struct_eval"

TASK_MARKDOWN = "markdown_extraction"
TASK_JSON = "json_extraction"
TASK_FORMULA = "formula_ocr"

STATUS_SUCCESS = "success"
STATUS_NA = "na"
STATUS_ERROR = "error"

MODEL_DOTS = "dots.ocr-1.5"
MODEL_LIGHTON = "LightOnOCR-2-1B"
MODEL_PADDLE_VL = "PaddleOCR-VL-1.5"
MODEL_PP_LAYOUT = "PP-DocLayoutV3_safetensors"
MODEL_ZHEN_LATEX = "ZhEn-Latex-OCR"


@dataclass
class EvalSample:
    dataset: str
    sample_id: str
    task: str
    ground_truth: str
    prompt: str
    image_loader: Callable[[], Image.Image]
    metadata: Optional[Dict[str, Any]] = None


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def text_score(pred: str, gt: str) -> float:
    pred_n = normalize_text(pred)
    gt_n = normalize_text(gt)
    denom = max(len(gt_n), 1)
    dist = levenshtein_distance(pred_n, gt_n)
    return max(0.0, min(1.0, 1.0 - dist / denom))


def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if blocks:
            return blocks[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def flatten_json(value: Any, prefix: str = "") -> List[str]:
    items: List[str] = []
    if isinstance(value, dict):
        for k in sorted(value.keys()):
            key = f"{prefix}.{k}" if prefix else str(k)
            items.extend(flatten_json(value[k], key))
    elif isinstance(value, list):
        for item in value:
            key = f"{prefix}[]" if prefix else "[]"
            items.extend(flatten_json(item, key))
    else:
        val = normalize_text(str(value))
        items.append(f"{prefix}={val}")
    return items


def json_score(pred: str, gt: str) -> Tuple[float, bool]:
    pred_block = extract_json_block(pred)
    gt_block = extract_json_block(gt)
    if not gt_block:
        return 0.0, False
    try:
        gt_obj = json.loads(gt_block)
    except Exception:
        return 0.0, False
    try:
        pred_obj = json.loads(pred_block if pred_block else pred)
        parse_ok = True
    except Exception:
        return 0.0, False

    gt_flat = Counter(flatten_json(gt_obj))
    pred_flat = Counter(flatten_json(pred_obj))
    if not gt_flat and not pred_flat:
        return 1.0, parse_ok
    if not gt_flat:
        return 0.0, parse_ok

    overlap = sum((gt_flat & pred_flat).values())
    p = overlap / max(sum(pred_flat.values()), 1)
    r = overlap / max(sum(gt_flat.values()), 1)
    if p + r == 0:
        return 0.0, parse_ok
    return 2 * p * r / (p + r), parse_ok


def default_prompt(task: str) -> str:
    if task == TASK_MARKDOWN:
        return "Extract all content from this document as clean markdown."
    if task == TASK_JSON:
        return "Extract the document into structured JSON."
    if task == TASK_FORMULA:
        return "Recognize the mathematical expression and output LaTeX only."
    return "Extract text."


def model_prompt(model_name: str, task: str) -> str:
    if model_name == MODEL_PADDLE_VL:
        if task == TASK_FORMULA:
            return "Formula Recognition:"
        if task == TASK_MARKDOWN:
            return "OCR:"
        if task == TASK_JSON:
            return "Table Recognition:"
    return default_prompt(task)


def safe_open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_ocr_benchmark_samples(root: Path) -> List[EvalSample]:
    meta_path = root / DATASET_OCR_BENCH / "test" / "metadata.jsonl"
    image_root = root / DATASET_OCR_BENCH / "test"
    samples: List[EvalSample] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image_rel = row.get("file_name")
            if not image_rel:
                continue
            image_path = image_root / image_rel
            sample_base = f"{row.get('id', image_rel)}"
            if row.get("true_markdown_output"):
                samples.append(
                    EvalSample(
                        dataset=DATASET_OCR_BENCH,
                        sample_id=f"{sample_base}:markdown",
                        task=TASK_MARKDOWN,
                        ground_truth=row["true_markdown_output"],
                        prompt=default_prompt(TASK_MARKDOWN),
                        image_loader=lambda p=image_path: safe_open_image(p),
                        metadata={"format": row.get("metadata")},
                    )
                )
            if row.get("true_json_output"):
                samples.append(
                    EvalSample(
                        dataset=DATASET_OCR_BENCH,
                        sample_id=f"{sample_base}:json",
                        task=TASK_JSON,
                        ground_truth=row["true_json_output"],
                        prompt=default_prompt(TASK_JSON),
                        image_loader=lambda p=image_path: safe_open_image(p),
                        metadata={"format": row.get("metadata")},
                    )
                )
    return samples


def load_hme100k_samples(root: Path) -> List[EvalSample]:
    zip_path = root / DATASET_HME100K / "test.zip"
    samples: List[EvalSample] = []
    with zipfile.ZipFile(zip_path) as zf:
        lines = zf.read("test_labels.txt").decode("utf-8", errors="replace").splitlines()
    for line in lines:
        if "\t" not in line:
            continue
        name, label = line.split("\t", 1)
        archive_name = f"test_images/{name}"

        def _loader(zp=zip_path, an=archive_name) -> Image.Image:
            with zipfile.ZipFile(zp) as zf:
                with zf.open(an) as fp:
                    return Image.open(BytesIO(fp.read())).convert("RGB")

        samples.append(
            EvalSample(
                dataset=DATASET_HME100K,
                sample_id=name,
                task=TASK_FORMULA,
                ground_truth=label,
                prompt=default_prompt(TASK_FORMULA),
                image_loader=_loader,
            )
        )
    return samples


def decode_latex_image(cell: Any, root: Path) -> Image.Image:
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(BytesIO(cell["bytes"])).convert("RGB")
        if cell.get("path"):
            p = Path(cell["path"])
            if not p.is_absolute():
                p = root / p
            return Image.open(p).convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(BytesIO(cell)).convert("RGB")
    if isinstance(cell, str):
        p = Path(cell)
        if not p.is_absolute():
            p = root / p
        return Image.open(p).convert("RGB")
    raise ValueError("Unsupported image field format for LaTeX_OCR.")


def load_latex_ocr_samples(root: Path) -> List[EvalSample]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for LaTeX_OCR evaluation.") from exc

    parquet_path = root / DATASET_LATEX_OCR / "data" / "test-00000-of-00001.parquet"
    table = pq.read_table(parquet_path, columns=["image", "text"])
    rows = table.to_pylist()
    samples: List[EvalSample] = []
    for i, row in enumerate(rows):
        text = row.get("text")
        image_cell = row.get("image")
        if not text:
            continue

        def _loader(c=image_cell, r=root / DATASET_LATEX_OCR) -> Image.Image:
            return decode_latex_image(c, r)

        samples.append(
            EvalSample(
                dataset=DATASET_LATEX_OCR,
                sample_id=f"latex_{i}",
                task=TASK_FORMULA,
                ground_truth=text,
                prompt=default_prompt(TASK_FORMULA),
                image_loader=_loader,
            )
        )
    return samples


def tc11_placeholder_samples() -> List[EvalSample]:
    return []


class BaseAdapter:
    supported_tasks: Tuple[str, ...] = ()

    def supports(self, task: str) -> bool:
        return task in self.supported_tasks

    def predict(self, image: Image.Image, prompt: str, task: str) -> str:
        raise NotImplementedError


class UnsupportedAdapter(BaseAdapter):
    supported_tasks: Tuple[str, ...] = ()


class GenericTransformerAdapter(BaseAdapter):
    supported_tasks = (TASK_MARKDOWN, TASK_JSON, TASK_FORMULA)

    def __init__(self, model_path: Path, device: str = "auto", max_new_tokens: int = 512, model_name: str = ""):
        self.model_path = str(model_path)
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.device = self._select_device(device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self._load_model()
        self.model.eval()

    @staticmethod
    def _select_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _model_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.device == "cuda" else torch.float32

    def _load_model(self):
        dtype = self._model_dtype()
        common_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if self.model_name == MODEL_PADDLE_VL:
            common_kwargs["attn_implementation"] = "sdpa"
        loaders = [
            AutoModelForImageTextToText,
            AutoModelForCausalLM,
        ]
        last_error = None
        for loader in loaders:
            try:
                model = loader.from_pretrained(self.model_path, **common_kwargs)
                model.to(self.device)
                return model
            except Exception as exc:
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError("Unable to load model.")

    def _to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def predict(self, image: Image.Image, prompt: str, task: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = None
        input_length = 0
        if hasattr(self.processor, "apply_chat_template"):
            try:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
            except Exception:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
        if inputs is None:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        if "input_ids" in inputs:
            input_length = int(inputs["input_ids"].shape[-1])
        inputs = self._to_device(inputs)

        # 尝试使用所有参数，如果失败则过滤掉 mm_token_type_ids
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        except (TypeError, ValueError) as e:
            if "mm_token_type_ids" in str(e):
                # 过滤掉不支持的参数
                filtered_inputs = {k: v for k, v in inputs.items() if k != "mm_token_type_ids"}
                with torch.no_grad():
                    output_ids = self.model.generate(**filtered_inputs, max_new_tokens=self.max_new_tokens)
            else:
                raise

        if output_ids.ndim == 2 and input_length > 0:
            output_ids = output_ids[:, input_length:]
        if hasattr(self.processor, "batch_decode"):
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return str(text).strip()


class ZhEnLatexAdapter(BaseAdapter):
    supported_tasks = (TASK_FORMULA,)

    def __init__(self, model_path: Path, device: str = "auto", max_new_tokens: int = 512):
        self.model_path = str(model_path)
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.feature_extractor = AutoImageProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, prompt: str, task: str) -> str:
        pixels = self.feature_extractor(image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            output = self.model.generate(pixels, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


def adapter_for_model(model_name: str, model_path: Path, device: str, max_new_tokens: int) -> BaseAdapter:
    if model_name == MODEL_PP_LAYOUT:
        return UnsupportedAdapter()
    if model_name == MODEL_ZHEN_LATEX:
        return ZhEnLatexAdapter(model_path, device=device, max_new_tokens=max_new_tokens)
    return GenericTransformerAdapter(model_path, device=device, max_new_tokens=max_new_tokens, model_name=model_name)


def score_prediction(task: str, pred: str, gt: str) -> Tuple[float, Dict[str, Any]]:
    if task in (TASK_MARKDOWN, TASK_FORMULA):
        return text_score(pred, gt), {"json_parse_ok": None}
    if task == TASK_JSON:
        score, parsed = json_score(pred, gt)
        return score, {"json_parse_ok": parsed}
    return 0.0, {"json_parse_ok": None}


def select_samples(samples: List[EvalSample], mode: str, smoke_samples: int, max_samples_per_dataset: int = 0) -> List[EvalSample]:
    grouped: Dict[str, List[EvalSample]] = defaultdict(list)
    for s in samples:
        grouped[s.dataset].append(s)
    selected: List[EvalSample] = []

    if mode == "full":
        # If max_samples_per_dataset is set, limit each dataset
        if max_samples_per_dataset > 0:
            for ds, items in grouped.items():
                del ds
                selected.extend(items[:max_samples_per_dataset])
        else:
            return samples
    else:
        # Smoke mode
        for ds, items in grouped.items():
            del ds
            selected.extend(items[:smoke_samples])

    return selected


def summarize(scores: List[float]) -> Optional[float]:
    if not scores:
        return None
    return float(statistics.mean(scores))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_outputs(summary_df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    ensure_dir(figures_dir)
    generated: List[Path] = []

    scored = summary_df.dropna(subset=["mean_score"]).copy()
    if scored.empty:
        return generated

    pivot = scored.pivot_table(
        index="model",
        columns="task",
        values="mean_score",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(pivot.columns)), max(4, 0.6 * len(pivot.index))))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Model x Task Score Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score")
    heatmap_path = figures_dir / "score_heatmap.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    generated.append(heatmap_path)

    for task, task_df in scored.groupby("task"):
        task_sorted = task_df.groupby("model", as_index=False)["mean_score"].mean().sort_values("mean_score", ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(task_sorted))))
        ax.bar(task_sorted["model"], task_sorted["mean_score"], color="#2E86AB")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_xlabel("model")
        ax.set_title(f"Task Scores - {task}")
        ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(task_sorted["mean_score"]):
            ax.text(i, min(v + 0.02, 1.0), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        path = figures_dir / f"task_bars_{task}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(path)

    return generated


def write_report(
    report_path: Path,
    mode: str,
    summary_df: pd.DataFrame,
    figure_paths: List[Path],
    output_csv: Path,
    output_json: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- mode: {mode}")
    lines.append(f"- summary_csv: `{output_csv.as_posix()}`")
    lines.append(f"- summary_json: `{output_json.as_posix()}`")
    lines.append("")

    if summary_df.empty:
        lines.append("No summary data produced.")
    else:
        lines.append("## Task-level scores")
        lines.append("")
        shown = summary_df.copy()
        shown["mean_score"] = shown["mean_score"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        shown["json_parse_rate"] = shown["json_parse_rate"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        try:
            table_text = shown.to_markdown(index=False)
        except Exception:
            table_text = "```csv\n" + shown.to_csv(index=False) + "\n```"
        lines.append(table_text)
        lines.append("")

    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for p in figure_paths:
            rel = p.as_posix()
            lines.append(f"![{p.name}]({rel})")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model x dataset task evaluation and generate score charts.")
    parser.add_argument("--data-root", default="data/ocrbenchmark")
    parser.add_argument("--model-root", default="data/model")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--smoke-samples", type=int, default=20)
    parser.add_argument("--max-samples-per-dataset", type=int, default=0, help="Max samples per dataset in full mode (0=unlimited)")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--detail-limit", type=int, default=200)
    parser.add_argument("--models", default="")
    parser.add_argument("--datasets", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir(DEFAULT_HF_HOME)
    ensure_dir(DEFAULT_HF_HOME / "hub")
    ensure_dir(DEFAULT_HF_HOME / "modules")

    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)
    docs_dir = Path(args.docs_dir)
    figures_dir = docs_dir / "figures"

    ensure_dir(output_dir)
    ensure_dir(docs_dir)
    ensure_dir(figures_dir)

    dataset_loaders = {
        DATASET_OCR_BENCH: lambda: load_ocr_benchmark_samples(data_root),
        DATASET_HME100K: lambda: load_hme100k_samples(data_root),
        DATASET_LATEX_OCR: lambda: load_latex_ocr_samples(data_root),
        DATASET_TC11: tc11_placeholder_samples,
    }

    selected_datasets = list(dataset_loaders.keys())
    if args.datasets.strip():
        requested = [x.strip() for x in args.datasets.split(",") if x.strip()]
        selected_datasets = [d for d in requested if d in dataset_loaders]

    all_samples: List[EvalSample] = []
    for ds in selected_datasets:
        loaded = dataset_loaders[ds]()
        all_samples.extend(loaded)
    all_samples = select_samples(all_samples, args.mode, args.smoke_samples, args.max_samples_per_dataset)

    model_names = [p.name for p in model_root.iterdir() if p.is_dir()]
    if args.models.strip():
        requested_models = [x.strip() for x in args.models.split(",") if x.strip()]
        model_names = [m for m in model_names if m in requested_models]
    model_names.sort()

    summary_rows: List[Dict[str, Any]] = []
    detail_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "details": {},
    }

    samples_by_dataset_task: Dict[Tuple[str, str], List[EvalSample]] = defaultdict(list)
    for sample in all_samples:
        samples_by_dataset_task[(sample.dataset, sample.task)].append(sample)

    for model_name in model_names:
        model_path = model_root / model_name
        print(f"[INFO] Loading model adapter: {model_name}")
        try:
            adapter = adapter_for_model(model_name, model_path, args.device, args.max_new_tokens)
            adapter_error = None
        except Exception as exc:
            adapter = UnsupportedAdapter()
            adapter_error = str(exc)
            print(f"[WARN] adapter init failed for {model_name}: {exc}")

        for (dataset, task), samples in samples_by_dataset_task.items():
            key = f"{model_name}|{dataset}|{task}"
            detail_payload["details"][key] = []
            if not adapter.supports(task):
                summary_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "task": task,
                        "mean_score": None,
                        "samples": len(samples),
                        "success_count": 0,
                        "error_count": 0,
                        "na_count": len(samples),
                        "json_parse_rate": None,
                        "note": "not_supported" if adapter_error is None else f"adapter_error: {adapter_error}",
                    }
                )
                continue

            scores: List[float] = []
            parse_flags: List[bool] = []
            success_count = 0
            error_count = 0
            for sample in samples:
                try:
                    prompt = model_prompt(model_name, task)
                    image = sample.image_loader()
                    pred = adapter.predict(image=image, prompt=prompt, task=task)
                    score, extras = score_prediction(task, pred, sample.ground_truth)
                    scores.append(score)
                    success_count += 1
                    if extras.get("json_parse_ok") is not None:
                        parse_flags.append(bool(extras["json_parse_ok"]))
                    if len(detail_payload["details"][key]) < args.detail_limit:
                        detail_payload["details"][key].append(
                            {
                                "sample_id": sample.sample_id,
                                "status": STATUS_SUCCESS,
                                "score": score,
                                "prediction_preview": pred[:300],
                            }
                        )
                except Exception as exc:
                    error_count += 1
                    scores.append(0.0)
                    if len(detail_payload["details"][key]) < args.detail_limit:
                        detail_payload["details"][key].append(
                            {
                                "sample_id": sample.sample_id,
                                "status": STATUS_ERROR,
                                "score": 0.0,
                                "error": str(exc),
                            }
                        )

            json_parse_rate = None
            if parse_flags:
                json_parse_rate = float(sum(1 for x in parse_flags if x) / len(parse_flags))

            summary_rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "task": task,
                    "mean_score": summarize(scores),
                    "samples": len(samples),
                    "success_count": success_count,
                    "error_count": error_count,
                    "na_count": 0,
                    "json_parse_rate": json_parse_rate,
                    "note": "",
                }
            )

        if DATASET_TC11 in selected_datasets:
            summary_rows.append(
                {
                    "model": model_name,
                    "dataset": DATASET_TC11,
                    "task": TASK_TC11_LEGACY,
                    "mean_score": None,
                    "samples": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "na_count": 0,
                    "json_parse_rate": None,
                    "note": "pending_legacy_eval",
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["dataset", "task", "model"]).reset_index(drop=True)
    csv_path = output_dir / "eval_results.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    json_path = output_dir / "eval_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary_rows,
                "detail_limit": args.detail_limit,
                "details": detail_payload["details"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    figure_paths = plot_outputs(summary_df, figures_dir)
    report_path = docs_dir / "benchmark_report.md"
    write_report(report_path, args.mode, summary_df, figure_paths, csv_path, json_path)

    print(f"[DONE] summary csv: {csv_path}")
    print(f"[DONE] summary json: {json_path}")
    print(f"[DONE] report: {report_path}")
    for p in figure_paths:
        print(f"[DONE] figure: {p}")


if __name__ == "__main__":
    main()
