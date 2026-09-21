from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


def train_description_model(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 40,
    patience: int = 8,
    max_text_length: int = 640,
    image_size: int = 192,
    hidden_size: int = 384,
    batch_size: int = 8,
    progress: Callable[[str], None] = print,
) -> dict:
    """Train a compact image-and-facies-to-text head on the short-description field."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Для обучения текста установите PyTorch.") from exc

    dataset_dir = Path(dataset_dir).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve(strict=True)
    caption_path = dataset_dir / "caption_dataset.jsonl"
    if not caption_path.is_file():
        raise ValueError("В датасете нет caption_dataset.jsonl с целями «Краткое описание».")
    rows = [json.loads(line) for line in caption_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = [row for row in rows if row.get("split") == "train" and row.get("target_text", "").strip()]
    val_rows = [row for row in rows if row.get("split") == "val" and row.get("target_text", "").strip()]
    if len(train_rows) < 5 or not val_rows:
        raise ValueError(
            "Для модели описания нужно минимум 5 обучающих фрагментов и хотя бы 1 независимый val-фрагмент с кратким описанием."
        )
    if epochs < 1 or patience < 1:
        raise ValueError("epochs и patience должны быть положительными.")
    output_path = output_dir / "description_best.pt"
    if output_path.exists():
        raise FileExistsError(f"Файл уже существует: {output_path}")

    characters = sorted({character for row in train_rows for character in _clean_text(row["target_text"])})
    tokens = list(SPECIAL_TOKENS) + characters
    token_to_id = {token: index for index, token in enumerate(tokens)}
    pad_id, bos_id, eos_id, unk_id = (token_to_id[token] for token in SPECIAL_TOKENS)
    facies_names = sorted({str(row.get("facies", "")).strip() for row in train_rows if str(row.get("facies", "")).strip()})
    facies_to_id = {name: index + 1 for index, name in enumerate(facies_names)}

    class CaptionDataset(Dataset):
        def __init__(self, items: list[dict]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            item = self.items[index]
            image = _read_crop(dataset_dir / item["crop"], image_size)
            text = _clean_text(item["target_text"])
            encoded = [bos_id] + [token_to_id.get(char, unk_id) for char in text[: max_text_length - 2]] + [eos_id]
            sequence = np.full(max_text_length, pad_id, dtype=np.int64)
            sequence[: len(encoded)] = encoded
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
            facies_id = facies_to_id.get(str(item.get("facies", "")).strip(), 0)
            return image_tensor, torch.tensor(facies_id, dtype=torch.long), torch.from_numpy(sequence)

    class DescriptionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                _conv_block(nn, 3, 32), _conv_block(nn, 32, 64),
                _conv_block(nn, 64, 128), _conv_block(nn, 128, 256),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, hidden_size), nn.Tanh(),
            )
            self.facies_embedding = nn.Embedding(len(facies_names) + 1, hidden_size)
            self.embedding = nn.Embedding(len(tokens), hidden_size, padding_idx=pad_id)
            self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, len(tokens))

        def forward(self, images, facies_ids, input_tokens):
            image_state = torch.tanh(self.encoder(images) + self.facies_embedding(facies_ids)).unsqueeze(0)
            decoded, _ = self.decoder(self.embedding(input_tokens), image_state)
            return self.output(decoded)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DescriptionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    train_loader = DataLoader(CaptionDataset(train_rows), batch_size=max(1, batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(CaptionDataset(val_rows), batch_size=max(1, batch_size), shuffle=False, num_workers=0)
    best_loss = math.inf
    stale_epochs = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(torch, model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(torch, model, val_loader, criterion, device, None)
        progress(f"Краткое описание: эпоха {epoch}/{epochs}, train loss={train_loss:.4f}, val loss={val_loss:.4f}")
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "schema": "excel-photo-description-v2",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model_state": model.state_dict(), "tokens": tokens,
                "max_text_length": max_text_length, "image_size": image_size,
                "hidden_size": hidden_size, "target_column": 22,
                "facies_names": facies_names,
                "conditioning": ["interval_image", "facies_class"],
                "target_header": "Краткое описание", "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "train_samples": len(train_rows), "val_samples": len(val_rows),
            }, output_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                progress(f"Краткое описание: ранняя остановка после эпохи {epoch}.")
                break
    if not output_path.is_file():
        raise RuntimeError("Обучение текста завершилось без description_best.pt.")
    info = {
        "schema": "excel-photo-description-training-v1", "model": str(output_path),
        "target_column": 22, "target_header": "Краткое описание",
        "train_samples": len(train_rows), "val_samples": len(val_rows),
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "device": str(device), "max_text_length": max_text_length,
        "facies_classes": facies_names,
        "warning": "Модель формирует текст только по визуально различимым признакам; факты, не видимые на фото, требуют проверки геолога.",
    }
    (output_dir / "description_training_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return info


def generate_description(model_path: Path, image_path: Path, facies: str = "") -> str:
    """Generate a short-description draft from one segmented interval crop."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Для применения модели установите PyTorch.") from exc
    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    schema = checkpoint.get("schema")
    if schema not in {"excel-photo-description-v1", "excel-photo-description-v2"}:
        raise ValueError("Неизвестный формат модели описания.")
    tokens = checkpoint["tokens"]
    token_to_id = {token: index for index, token in enumerate(tokens)}
    pad_id, bos_id, eos_id = (token_to_id[token] for token in SPECIAL_TOKENS[:3])
    hidden_size = int(checkpoint["hidden_size"])
    image_size = int(checkpoint["image_size"])
    facies_names = list(checkpoint.get("facies_names", []))
    facies_to_id = {name: index + 1 for index, name in enumerate(facies_names)}

    class DescriptionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                _conv_block(nn, 3, 32), _conv_block(nn, 32, 64),
                _conv_block(nn, 64, 128), _conv_block(nn, 128, 256),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, hidden_size), nn.Tanh(),
            )
            if schema == "excel-photo-description-v2":
                self.facies_embedding = nn.Embedding(len(facies_names) + 1, hidden_size)
            self.embedding = nn.Embedding(len(tokens), hidden_size, padding_idx=pad_id)
            self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, len(tokens))

    model = DescriptionNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image = torch.from_numpy(_read_crop(Path(image_path), image_size).transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        image_state = model.encoder(image)
        if schema == "excel-photo-description-v2":
            facies_id = torch.tensor([facies_to_id.get(str(facies).strip(), 0)], dtype=torch.long)
            image_state = torch.tanh(image_state + model.facies_embedding(facies_id))
        state = image_state.unsqueeze(0)
        current = torch.tensor([[bos_id]], dtype=torch.long)
        output = []
        for _ in range(int(checkpoint["max_text_length"]) - 1):
            decoded, state = model.decoder(model.embedding(current), state)
            next_id = int(model.output(decoded[:, -1]).argmax(dim=-1).item())
            if next_id == eos_id:
                break
            if next_id not in {pad_id, bos_id}:
                output.append(tokens[next_id])
            current = torch.tensor([[next_id]], dtype=torch.long)
    return "".join(output).strip()


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
    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, facies_ids, sequence in loader:
            images = images.to(device)
            facies_ids = facies_ids.to(device)
            sequence = sequence.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, facies_ids, sequence[:, :-1])
            loss = criterion(logits.reshape(-1, logits.shape[-1]), sequence[:, 1:].reshape(-1))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
            total_items += images.shape[0]
    return total_loss / max(1, total_items)


def _read_crop(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось открыть вырезку: {path}")
    image = cv2.cvtColor(cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
