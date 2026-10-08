from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .paths import resolve_existing_path


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
MODEL_SCHEMA = "excel-photo-description-v5"
MIN_TRAIN_SAMPLES = 20
MIN_VAL_SAMPLES = 3


def train_description_model(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 40,
    patience: int = 8,
    image_size: int = 128,
    hidden_size: int = 128,
    batch_size: int = 8,
    device: str | int = 0,
    progress: Callable[[str], None] = print,
) -> dict:
    """Train a small interval-image + facies-conditioned character decoder."""
    dataset_dir = resolve_existing_path(dataset_dir)
    output_dir = Path(output_dir).expanduser().absolute()
    caption_path = dataset_dir / "caption_dataset.jsonl"
    if not caption_path.is_file():
        return _not_trained("В датасете нет caption_dataset.jsonl с привязкой масок к Excel-описаниям.")
    rows = [
        json.loads(line) for line in caption_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [row for row in rows if row.get("split") == "train" and _clean_text(row.get("target_text"))]
    val_rows = [row for row in rows if row.get("split") == "val" and _clean_text(row.get("target_text"))]
    if len(train_rows) < MIN_TRAIN_SAMPLES or len(val_rows) < MIN_VAL_SAMPLES:
        return _not_trained(
            f"Для генерации описаний нужно минимум {MIN_TRAIN_SAMPLES} обучающих и "
            f"{MIN_VAL_SAMPLES} независимых проверочных интервала; добавьте и подтвердите ещё скважины."
        )
    if epochs < 1 or patience < 1 or image_size < 32 or hidden_size < 16 or batch_size < 1:
        raise ValueError("Параметры обучения текстовой модели должны быть положительными; image_size >= 32, hidden_size >= 16.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Папка текстовой модели не пуста: {output_dir}")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Для обучения генерации текста нужна CUDA-сборка PyTorch.") from exc
    if isinstance(device, int) or str(device).strip().isdigit():
        device = f"cuda:{device}"
    device = str(device).strip().casefold()
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("Устройство текстовой модели — cuda:0; явное cpu разрешено только для тестов.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU недоступна для обучения генерации краткого описания.")
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

    characters = sorted({character for row in train_rows for character in _clean_text(row["target_text"])})
    tokens = list(SPECIAL_TOKENS) + characters
    token_to_id = {token: index for index, token in enumerate(tokens)}
    pad_id, bos_id, eos_id, unk_id = (token_to_id[token] for token in SPECIAL_TOKENS)
    facies_conditions = sorted({
        (
            str(row.get("facies_index") or row.get("facies") or "").strip(),
            str(row.get("facies_name") or row.get("facies_index") or row.get("facies") or "").strip(),
        )
        for row in train_rows + val_rows
        if str(row.get("facies_index") or row.get("facies") or "").strip()
    }, key=lambda item: (item[0].casefold(), item[1].casefold()))
    facies_to_id = {_facies_key(index, name): position + 1 for position, (index, name) in enumerate(facies_conditions)}
    facies_indices = [index for index, _name in facies_conditions]
    max_text_length = max(len(_clean_text(row["target_text"])) + 2 for row in train_rows + val_rows)
    if max_text_length > 2048:
        raise ValueError("Описание длиннее 2046 символов; проверьте, что Excel-колонка содержит именно краткое описание.")

    class CaptionDataset(Dataset):
        def __init__(self, items: list[dict]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            item = self.items[index]
            crop_path = dataset_dir / str(item["crop"])
            image = _prepare_crop(crop_path, image_size)
            text = _clean_text(item["target_text"])
            encoded = [bos_id] + [token_to_id.get(char, unk_id) for char in text] + [eos_id]
            facies = str(item.get("facies_index") or item.get("facies") or "").strip()
            facies_name = str(item.get("facies_name") or facies).strip()
            thickness = float(item.get("interval_m", 0.0) or 0.0)
            return (
                torch.from_numpy(image.transpose(2, 0, 1)),
                torch.tensor(facies_to_id.get(_facies_key(facies, facies_name), 0), dtype=torch.long),
                torch.tensor([math.log1p(max(0.0, thickness))], dtype=torch.float32),
                torch.tensor(encoded, dtype=torch.long),
            )

    def collate_batch(items):
        images, facies_ids, thicknesses, sequences = zip(*items)
        return (
            torch.stack(images), torch.stack(facies_ids), torch.stack(thicknesses),
            nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_id),
        )

    class DescriptionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                _conv_block(nn, 3, 24), _conv_block(nn, 24, 48), _conv_block(nn, 48, 96),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(96, hidden_size), nn.Tanh(),
            )
            self.facies_embedding = nn.Embedding(len(facies_conditions) + 1, hidden_size)
            self.interval_embedding = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh())
            self.embedding = nn.Embedding(len(tokens), hidden_size, padding_idx=pad_id)
            self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, len(tokens))

        def forward(self, images, facies_ids, thicknesses, input_tokens):
            initial_state = torch.tanh(
                self.encoder(images) + self.facies_embedding(facies_ids)
                + self.interval_embedding(thicknesses)
            ).unsqueeze(0)
            decoded, _ = self.decoder(self.embedding(input_tokens), initial_state)
            return self.output(decoded)

    torch.manual_seed(42)
    model = DescriptionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    train_loader = DataLoader(
        CaptionDataset(train_rows), batch_size=max(1, batch_size), shuffle=True,
        num_workers=0, collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        CaptionDataset(val_rows), batch_size=max(1, batch_size), shuffle=False,
        num_workers=0, collate_fn=collate_batch,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "description_best.pt"
    best_loss, stale_epochs, best_epoch = math.inf, 0, 0
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(torch, model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(torch, model, val_loader, criterion, device, None)
        progress(f"Генерация описания: эпоха {epoch}/{epochs}, train loss={train_loss:.4f}, val loss={val_loss:.4f}")
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save({
                "schema": MODEL_SCHEMA, "created_at": datetime.now().isoformat(timespec="seconds"),
                "model_state": model.state_dict(), "tokens": tokens,
                "max_text_length": max_text_length, "image_size": image_size,
                "hidden_size": hidden_size,
                "facies_conditions": [
                    {"facies_index": index, "facies_name": name}
                    for index, name in facies_conditions
                ],
                "facies_catalog": _facies_catalog(dataset_dir, train_rows + val_rows),
                "target_fields": ["facies_index", "facies_name", "target_text"],
                "target_headers": {
                    "facies_index": "Индекс фации", "facies_name": "Название фации",
                    "target_text": "Краткое описание",
                },
                "conditioning": ["interval_image_crop", "facies_index", "facies_name", "interval_thickness_m"],
                "best_epoch": best_epoch, "best_val_loss": best_loss,
                "train_samples": len(train_rows), "val_samples": len(val_rows),
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                progress(f"Генерация описания: ранняя остановка после эпохи {epoch}.")
                break
    if not checkpoint_path.is_file():
        raise RuntimeError("Обучение генерации описания завершилось без description_best.pt.")
    info = {
        "schema": "excel-photo-description-training-v2", "status": "trained_candidate",
        "model": checkpoint_path.name, "target_header": "Краткое описание",
        "target_headers": {
            "facies_index": "Индекс фации", "facies_name": "Название фации",
            "target_text": "Краткое описание",
        },
        "conditioning": ["interval_image_crop", "facies_index", "facies_name", "interval_thickness_m"],
        "train_samples": len(train_rows), "val_samples": len(val_rows),
        "best_epoch": best_epoch, "best_val_loss": best_loss, "device": device,
        "facies_classes": facies_indices,
        "warning": "Текст — нейросетевой черновик по интервалу и подтверждённым описаниям; перед геологическим использованием проверьте его.",
    }
    (output_dir / "description_training_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    progress(f"Модель описаний сохранена: {checkpoint_path}")
    return info


class DescriptionGenerator:
    """Load one interval-conditioned description network and reuse it for a well."""

    def __init__(self, model_path: Path | dict):
        import torch
        checkpoint = model_path if isinstance(model_path, dict) else torch.load(
            Path(model_path), map_location="cpu", weights_only=False,
        )
        self.torch = torch
        self.model, self.checkpoint = _load_description_network(checkpoint)

    def generate_with_confidence(
        self, image: Path | np.ndarray, facies: str = "", facies_name: str = "",
        interval_m: float = 0.0,
    ) -> tuple[str, float]:
        torch = self.torch
        checkpoint = self.checkpoint
        tokens = checkpoint["tokens"]
        token_to_id = {token: index for index, token in enumerate(tokens)}
        bos_id, eos_id = token_to_id["<bos>"], token_to_id["<eos>"]
        forbidden = [token_to_id[token] for token in ("<pad>", "<bos>", "<unk>")]
        facies_to_id = {
            _facies_key(item["facies_index"], item.get("facies_name", item["facies_index"])): index + 1
            for index, item in enumerate(checkpoint["facies_conditions"])
        }
        facies_to_id_by_index = {
            str(item["facies_index"]).strip().casefold(): index + 1
            for index, item in enumerate(checkpoint["facies_conditions"])
        }
        condition_id = facies_to_id.get(
            _facies_key(facies, facies_name or facies),
            facies_to_id_by_index.get(str(facies).strip().casefold(), 0),
        )
        facies_id = torch.tensor([condition_id], dtype=torch.long)
        thickness = torch.tensor([[math.log1p(max(0.0, float(interval_m)))]], dtype=torch.float32)
        image_tensor = torch.from_numpy(_prepare_crop(image, int(checkpoint["image_size"])).transpose(2, 0, 1)).float().unsqueeze(0)
        model = self.model
        with torch.inference_mode():
            state = torch.tanh(
                model.encoder(image_tensor) + model.facies_embedding(facies_id)
                + model.interval_embedding(thickness)
            ).unsqueeze(0)
            current = torch.tensor([[bos_id]], dtype=torch.long)
            output, token_confidences = [], []
            for _ in range(int(checkpoint["max_text_length"]) - 1):
                decoded, state = model.decoder(model.embedding(current), state)
                logits = model.output(decoded[:, -1])
                logits[:, forbidden] = -float("inf")
                probabilities = torch.softmax(logits, dim=-1)
                next_id = int(probabilities.argmax(dim=-1).item())
                if next_id == eos_id:
                    break
                token_confidences.append(float(probabilities[0, next_id].item()))
                output.append(tokens[next_id])
                current = torch.tensor([[next_id]], dtype=torch.long)
        text = _clean_text("".join(output))
        confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
        if not _valid_generated_text(text):
            return "", confidence
        return text, confidence

    def generate(
        self, image: Path | np.ndarray, facies: str = "", facies_name: str = "",
        interval_m: float = 0.0,
    ) -> str:
        return self.generate_with_confidence(image, facies, facies_name, interval_m)[0]


def _load_description_network(checkpoint: dict):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Для применения модели описания установите PyTorch.") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != MODEL_SCHEMA:
        raise ValueError("Неизвестный формат модели генерации описаний.")
    tokens, facies_conditions = checkpoint["tokens"], list(checkpoint["facies_conditions"])
    pad_id = tokens.index("<pad>")
    hidden_size, image_size = int(checkpoint["hidden_size"]), int(checkpoint["image_size"])

    class DescriptionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                _conv_block(nn, 3, 24), _conv_block(nn, 24, 48), _conv_block(nn, 48, 96),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(96, hidden_size), nn.Tanh(),
            )
            self.facies_embedding = nn.Embedding(len(facies_conditions) + 1, hidden_size)
            self.interval_embedding = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh())
            self.embedding = nn.Embedding(len(tokens), hidden_size, padding_idx=pad_id)
            self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, len(tokens))

    model = DescriptionNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _conv_block(nn, input_channels: int, output_channels: int):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
    )


def _run_epoch(torch, model, loader, criterion, device, optimizer):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, facies_ids, thicknesses, sequence in loader:
            images, facies_ids = images.to(device), facies_ids.to(device)
            thicknesses, sequence = thicknesses.to(device), sequence.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, facies_ids, thicknesses, sequence[:, :-1])
            loss = criterion(logits.reshape(-1, logits.shape[-1]), sequence[:, 1:].reshape(-1))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
            total_items += images.shape[0]
    return total_loss / max(1, total_items)


def _prepare_crop(source: Path | np.ndarray, image_size: int) -> np.ndarray:
    image = source if isinstance(source, np.ndarray) else cv2.imdecode(
        np.frombuffer(Path(source).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Не удалось открыть вырезку керна для модели краткого описания.")
    height, width = image.shape[:2]
    scale = min(image_size / max(1, width), image_size / max(1, height))
    resized = cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    y = (image_size - resized.shape[0]) // 2
    x = (image_size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _facies_catalog(dataset_dir: Path, rows: list[dict]) -> list[dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for row in rows:
        index = str(row.get("facies_index") or row.get("facies") or "").strip()
        if index:
            catalog.setdefault(index.casefold(), {
                "facies_index": index,
                "facies_name": str(row.get("facies_name") or index).strip(),
            })
    metadata_path = dataset_dir / "class_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for item in metadata.get("classes", []):
            index = str(item.get("facies_index", "")).strip()
            if index:
                catalog[index.casefold()] = {
                    "facies_index": index,
                    "facies_name": str(item.get("facies_name") or index),
                }
    return sorted(catalog.values(), key=lambda item: item["facies_index"].casefold())


def _clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _facies_key(index: str, name: str) -> tuple[str, str]:
    return _clean_text(index).casefold(), _clean_text(name).casefold()


def _valid_generated_text(value: str) -> bool:
    if len(value) < 4 or any(not character.isprintable() for character in value):
        return False
    if re.search(r"(.)\1{7,}", value) or "<unk>" in value or "<eos>" in value:
        return False
    return sum(character.isalpha() for character in value) >= 3


def _not_trained(reason: str) -> dict:
    return {"schema": "excel-photo-description-training-v2", "status": "not_trained", "reason": reason}
