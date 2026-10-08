from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .paths import resolve_existing_path


VLM_SCHEMA = "excel-photo-qwen3-vl-lora-v1"
MIN_TRAIN_SAMPLES = 20
MIN_VAL_SAMPLES = 3
DEFAULT_VLM_ADAPTER = (
    Path.home() / "Desktop" / "Синтетический керн" / "vlm_core_v1" / "outputs"
    / "core_vlm_v3_knowledge_lora"
)
DEFAULT_VLM_BASE = (
    Path.home() / "Desktop" / "Синтетический керн" / "vlm_core_v1" / "model_cache" / "hub"
    / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
    / "89644892e4d85e24eaac8bacfd4f463576704203"
)


def default_vlm_paths() -> tuple[Path | None, Path | None]:
    """Return the user's local pilot adapter/base when that known layout exists."""
    adapter = DEFAULT_VLM_ADAPTER if (DEFAULT_VLM_ADAPTER / "adapter_config.json").is_file() else None
    base = DEFAULT_VLM_BASE if (DEFAULT_VLM_BASE / "config.json").is_file() else None
    return base, adapter


def normalize_adapter_path(path: str | Path) -> Path:
    """Accept either an adapter folder or its adapter_model.safetensors file."""
    value = Path(path).expanduser()
    if value.is_file():
        if value.name.casefold() not in {"adapter_model.safetensors", "adapter_model.bin"}:
            raise ValueError("Выберите adapter_model.safetensors или папку LoRA-адаптера.")
        value = value.parent
    value = resolve_existing_path(value)
    if not (value / "adapter_config.json").is_file():
        raise FileNotFoundError(f"В папке LoRA не найден adapter_config.json: {value}")
    if not any((value / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"В папке LoRA не найдены веса adapter_model: {value}")
    return value


def validate_base_model_path(path: str | Path) -> Path:
    value = resolve_existing_path(path)
    if not (value / "config.json").is_file():
        raise FileNotFoundError(f"В папке базовой VLM-модели не найден config.json: {value}")
    has_weights = any((value / name).is_file() for name in (
        "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ))
    if not has_weights:
        raise FileNotFoundError(f"В папке базовой VLM-модели не найдены локальные веса: {value}")
    return value


def build_description_prompt(
    *, facies_index: str, facies_name: str, interval_top: float, interval_base: float,
    interval_m: float, association: str = "", environment: str = "",
) -> str:
    """Prompt for text only: all interval geometry and class labels come from YOLO/metadata."""
    context = [
        f"Индекс фации (уже определён моделью сегментации): {facies_index or 'не указан'}.",
        f"Название фации из справочника: {facies_name or 'не указано'}.",
        f"Заданный интервал: {interval_top:.2f}–{interval_base:.2f} м; толщина {interval_m:.2f} м.",
    ]
    if association:
        context.append(f"Подтверждённая ассоциация из Excel: {association}.")
    if environment:
        context.append(f"Подтверждённая обстановка из Excel: {environment}.")
    return (
        "Составь краткое описание керна на русском языке только для заданного интервала. "
        "Границы интервала и индекс фации уже заданы отдельной моделью и не подлежат изменению. "
        "Опиши лишь признаки, которые действительно видны на изображении. Не выводи глубины, "
        "интервалы, названия или индексы классов, не угадывай минералы, зернистость, пористость, "
        "прочность, трещины, биотурбацию или геологическую обстановку. Если визуальных признаков "
        "недостаточно, так и укажи. Верни только 1–2 коротких предложения без JSON и заголовка.\n"
        + "\n".join(context)
    )


def clean_generated_description(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:описание|краткое описание)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    # Prevent a verbose generation from turning the Excel field into a report.
    return text[:600].strip()


def _read_caption_rows(dataset_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    caption_path = dataset_dir / "caption_dataset.jsonl"
    if not caption_path.is_file():
        raise ValueError("В датасете нет caption_dataset.jsonl с подтверждёнными интервалами и описаниями Excel.")
    rows = [
        json.loads(line) for line in caption_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    valid, skipped = [], []
    for row in rows:
        crop = dataset_dir / str(row.get("crop", ""))
        if row.get("split") not in {"train", "val"} or not str(row.get("target_text", "")).strip() or not crop.is_file():
            skipped.append(row)
            continue
        valid.append(row)
    return valid, [row for row in valid if row["split"] == "train"], [row for row in valid if row["split"] == "val"]


def train_vlm_description_model(
    dataset_dir: Path,
    output_dir: Path,
    *,
    base_model_dir: str | Path,
    initial_adapter: str | Path,
    epochs: float = 2.0,
    learning_rate: float = 1e-5,
    max_image_side: int = 448,
    device: str | int = "cuda:0",
    progress: Callable[[str], None] = print,
) -> dict:
    """Continue a local Qwen3-VL LoRA adapter on reviewed image/Excel description pairs."""
    dataset_dir = resolve_existing_path(dataset_dir)
    output_dir = Path(output_dir).expanduser().absolute()
    base_model_dir = validate_base_model_path(base_model_dir)
    initial_adapter = normalize_adapter_path(initial_adapter)
    if str(device).strip().casefold() not in {"0", "cuda", "cuda:0"}:
        raise ValueError("Дообучение VLM в этой версии запускается только на GPU CUDA 0.")
    if epochs <= 0 or learning_rate <= 0 or max_image_side < 64:
        raise ValueError("Проверьте эпохи, learning rate и размер изображения VLM.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Папка VLM-адаптера не пуста: {output_dir}")
    _rows, train_rows, val_rows = _read_caption_rows(dataset_dir)
    if len(train_rows) < MIN_TRAIN_SAMPLES or len(val_rows) < MIN_VAL_SAMPLES:
        return {
            "schema": VLM_SCHEMA, "status": "not_trained",
            "reason": (
                f"Для дообучения VLM нужно минимум {MIN_TRAIN_SAMPLES} обучающих и "
                f"{MIN_VAL_SAMPLES} независимых проверочных подтверждённых интервалов; "
                f"сейчас {len(train_rows)} и {len(val_rows)}. YOLO-сегментация обучается отдельно."
            ),
            "train_samples": len(train_rows), "val_samples": len(val_rows),
        }
    try:
        import torch
        from PIL import Image
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            "Для VLM нужны transformers, peft, accelerate, Pillow и safetensors. "
            "Установите дополнительные зависимости приложения (pip install -e .[vlm])."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU недоступна: обучение описаний остановлено без перехода на CPU.")
    try:
        torch.cuda.set_device(0)
        gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        raise RuntimeError(f"Не удалось инициализировать CUDA GPU 0 для VLM: {exc}") from exc

    class ReviewedCaptionDataset(torch.utils.data.Dataset):
        def __init__(self, samples: list[dict]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            row = self.samples[index]
            with Image.open(dataset_dir / row["crop"]) as opened:
                image = opened.convert("RGB")
                image.thumbnail((max_image_side, max_image_side))
            prompt = build_description_prompt(
                facies_index=str(row.get("facies_index", row.get("facies", ""))),
                facies_name=str(row.get("facies_name", "")),
                interval_top=float(row.get("interval_top", row.get("depth_top", 0))),
                interval_base=float(row.get("interval_base", row.get("depth_base", 0))),
                interval_m=float(row.get("interval_m", 0)),
                association=str(row.get("association", "")),
                environment=str(row.get("environment", "")),
            )
            user_message = {"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt},
            ]}
            answer = clean_generated_description(str(row["target_text"]))
            full_text = processor.apply_chat_template(
                [user_message, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                tokenize=False, add_generation_prompt=False,
            )
            prompt_text = processor.apply_chat_template(
                [user_message], tokenize=False, add_generation_prompt=True,
            )
            full = processor(text=[full_text], images=[image], return_tensors="pt")
            prompt_encoding = processor(text=[prompt_text], images=[image], return_tensors="pt")
            labels = full["input_ids"].clone()
            labels[:, :prompt_encoding["input_ids"].shape[1]] = -100
            if "attention_mask" in full:
                labels[full["attention_mask"] == 0] = -100
            return {
                key: value.squeeze(0) if key in {"input_ids", "attention_mask", "labels"} else value
                for key, value in {**full, "labels": labels}.items()
            }

    class SingleImageCollator:
        """Pad text and concatenate variable-resolution visual patches for batch size 1."""
        def __init__(self, pad_token_id: int):
            self.pad_token_id = int(pad_token_id)

        def __call__(self, features: list[dict]) -> dict:
            max_len = max(item["input_ids"].shape[-1] for item in features)
            batch = len(features)
            input_ids = torch.full((batch, max_len), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
            labels = torch.full((batch, max_len), -100, dtype=torch.long)
            for index, item in enumerate(features):
                size = item["input_ids"].shape[-1]
                input_ids[index, :size] = item["input_ids"]
                attention_mask[index, :size] = item["attention_mask"]
                labels[index, :size] = item["labels"]
            result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            common = set.intersection(*(set(item) for item in features)) - set(result)
            for key in common:
                tensors = [item[key] for item in features]
                result[key] = torch.cat(tensors, dim=0) if tensors[0].ndim else torch.stack(tensors)
            return result

    progress(f"Загружаю базовую Qwen3-VL из локальной папки: {base_model_dir}")
    processor = AutoProcessor.from_pretrained(str(base_model_dir), local_files_only=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_model_dir), torch_dtype=torch.float16, attn_implementation="sdpa",
        device_map={"": "cuda:0"}, local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, str(initial_adapter), is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    progress(
        f"Дообучаю LoRA VLM на GPU {gpu_name}: {len(train_rows)} train / "
        f"{len(val_rows)} независимых val интервалов; базовые веса не изменяются."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    import inspect
    training_signature = inspect.signature(TrainingArguments).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in training_signature else "evaluation_strategy"
    training_kwargs = {
        "output_dir": str(output_dir / "trainer_runs"),
        "num_train_epochs": float(epochs),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": float(learning_rate),
        "fp16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch",
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": 42,
    }
    training_kwargs[strategy_name] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": ReviewedCaptionDataset(train_rows),
        "eval_dataset": ReviewedCaptionDataset(val_rows),
        "data_collator": SingleImageCollator(processor.tokenizer.pad_token_id),
    }
    trainer_signature = __import__("inspect").signature(Trainer).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    trainer = Trainer(**trainer_kwargs)
    result = trainer.train()
    metrics = dict(getattr(result, "metrics", {}) or {})
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    try:
        shutil.rmtree(output_dir / "trainer_runs", ignore_errors=True)
    except OSError:
        pass
    pilot_warning = (
        "The supplied core_vlm_v3_knowledge_lora pilot contains only four image/text examples and its "
        "recorded qualitative check failed even on a training image. "
        if initial_adapter.name.casefold() == "core_vlm_v3_knowledge_lora" else ""
    )
    info = {
        "schema": VLM_SCHEMA,
        "status": "trained_candidate_requires_review",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model_dir": str(base_model_dir),
        "initial_adapter_dir": str(initial_adapter),
        "device": "cuda:0",
        "gpu_name": str(gpu_name),
        "train_samples": len(train_rows), "val_samples": len(val_rows),
        "train_loss": metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "epochs": float(epochs), "learning_rate": float(learning_rate),
        "conditioning": ["interval image crop", "facies index/name", "verified depth interval", "thickness"],
        "role": "draft Russian visual description only; YOLO owns segmentation, depth and facies class",
        "quality_note": (
            "This is an experimental candidate, not a validated production model. " + pilot_warning
            + "Inspect descriptions on independent wells before geological use."
        ),
    }
    (output_dir / "description_training_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    progress(f"VLM LoRA-адаптер сохранён: {output_dir / 'adapter_model.safetensors'}")
    return info


class VLMDescriptionGenerator:
    """Lazy GPU-only generator for short descriptions; it never predicts interval geometry."""

    def __init__(self, base_model_dir: str | Path, adapter_dir: str | Path, *, device: str = "cuda:0"):
        if device.casefold() not in {"cuda", "cuda:0"}:
            raise ValueError("Генерация описаний VLM в этой версии требует CUDA GPU 0.")
        self.base_model_dir = validate_base_model_path(base_model_dir)
        self.adapter_dir = normalize_adapter_path(adapter_dir)
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Для генерации VLM нужны transformers, peft, accelerate, Pillow и safetensors "
                "(pip install -e .[vlm])."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU недоступна для загрузки VLM; CPU-переход отключён.")
        from PIL import Image
        self._torch = torch
        self._Image = Image
        self._processor = AutoProcessor.from_pretrained(str(self.base_model_dir), local_files_only=True)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.base_model_dir), torch_dtype=torch.float16, attn_implementation="sdpa",
            device_map={"": "cuda:0"}, local_files_only=True,
        )
        self._model = PeftModel.from_pretrained(base, str(self.adapter_dir), is_trainable=False)
        self._model.eval()

    def generate(
        self, image, *, facies_index: str, facies_name: str, interval_top: float,
        interval_base: float, interval_m: float, association: str = "", environment: str = "",
        max_new_tokens: int = 128,
    ) -> str:
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if getattr(image, "ndim", 0) == 3 else image
        pil_image = Image.fromarray(rgb).convert("RGB")
        pil_image.thumbnail((448, 448))
        prompt = build_description_prompt(
            facies_index=facies_index, facies_name=facies_name,
            interval_top=interval_top, interval_base=interval_base, interval_m=interval_m,
            association=association, environment=environment,
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_image}, {"type": "text", "text": prompt},
        ]}]
        encoded_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[encoded_text], images=[pil_image], return_tensors="pt",
        ).to("cuda:0")
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completion = generated[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(completion, skip_special_tokens=True)[0]
        return clean_generated_description(text)
