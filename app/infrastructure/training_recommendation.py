"""Dataset-aware epoch recommendations for facies fine-tuning.

The recommendation is deliberately conservative.  Epoch count cannot repair
missing geological diversity, duplicated masks, or a class observed only once;
in those cases we lower the overfitting ceiling and report low confidence
instead of pretending that a larger number of epochs is more precise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.domain.models import PhotoRecord
from app.infrastructure.training_quality import TrainingQuality, training_quality


@dataclass(frozen=True)
class TrainingRecommendation:
    recommended_epochs: int
    max_epochs: int
    patience: int
    confidence: str
    verified_masks: int
    source_photos: int
    wells: int
    classes: int
    min_class_examples: int
    min_class_photos: int
    new_classes: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def short_text(self) -> str:
        return (
            f"ориентир: {self.recommended_epochs} · максимум: {self.max_epochs} · "
            f"автостоп: {self.patience}"
        )

    @property
    def details(self) -> str:
        lines = [
            f"Рекомендуемая рабочая точка: {self.recommended_epochs} эпох.",
            f"Безопасный максимум: {self.max_epochs} эпох.",
            f"Ранняя остановка: после {self.patience} эпох без улучшения.",
            f"Надёжность оценки: {self.confidence}.",
            (
                f"Учтено: масок {self.verified_masks}, исходных фото {self.source_photos}, "
                f"скважин {self.wells}, фаций {self.classes}, "
                f"минимум в одной фации {self.min_class_examples} масок / "
                f"{self.min_class_photos} фото."
            ),
        ]
        if self.new_classes:
            lines.append("Новые для выбранной модели фации: " + ", ".join(self.new_classes) + ".")
        if self.reasons:
            lines.extend(f"• {reason}" for reason in self.reasons)
        return "\n".join(lines)


def recommend_training(
    records: list[PhotoRecord],
    known_model_classes: set[str] | None = None,
    quality: TrainingQuality | None = None,
) -> TrainingRecommendation:
    """Estimate a useful epoch region from independent and correlated evidence.

    ``known_model_classes=None`` means that the selected model has no readable
    facies catalogue.  That is different from an explicitly empty catalogue:
    unknown classes must not all be reported as new.
    """
    quality = quality or training_quality(records)
    approved_records = [
        record
        for record in records
        if any(
            detection.training_ready
            and str(detection.label or "").strip() not in {"", "Новый контур"}
            and len(detection.polygon) >= 3
            for detection in record.detections
        )
    ]
    masks = int(quality.verified)
    photos = len(approved_records)
    wells = len({str(record.well_name or "Скважина 1").strip() for record in approved_records})
    class_counts = quality.by_facies
    classes = len(class_counts)
    min_class = min(class_counts.values(), default=0)
    photos_by_class: dict[str, set[int]] = defaultdict(set)
    for record in approved_records:
        for detection in record.detections:
            label = str(detection.label or "").strip()
            if detection.training_ready and label in class_counts and len(detection.polygon) >= 3:
                photos_by_class[label].add(id(record))
    min_class_photos = min((len(values) for values in photos_by_class.values()), default=0)
    density = masks / max(1, photos)

    if masks < 20:
        epochs = 60
    elif masks < 50:
        epochs = 50
    elif masks < 100:
        epochs = 40
    elif masks < 250:
        epochs = 32
    else:
        epochs = 24

    reasons: list[str] = []

    # More classes make the decision boundary harder even when the total
    # number of masks is unchanged.
    class_adjustment = min(15, max(0, classes - 2) * 2)
    if class_adjustment:
        epochs += class_adjustment
        reasons.append(f"{classes} фаций: добавлен запас на более сложное разделение классов")

    # Many neighbouring intervals cut from one photograph are correlated and
    # should not be treated as independent evidence.
    if photos <= 1 and masks:
        epochs -= 15
        reasons.append("все примеры с одного фото: максимум снижен против переобучения")
    elif photos <= 3 and masks:
        epochs -= 8
        reasons.append("мало независимых фото: максимум снижен против переобучения")
    elif photos >= 10:
        epochs += 5
        reasons.append("много независимых фото: модель может безопасно учиться дольше")
    if wells >= 2:
        epochs += 5
        reasons.append("несколько скважин: учтено дополнительное визуальное разнообразие")
    if density >= 8:
        epochs -= 5
        reasons.append("много соседних масок на одно фото: учтена их коррелированность")

    if min_class and min_class <= 2:
        epochs -= 5
        reasons.append("в редкой фации не больше двух примеров: лишние эпохи только запомнят их")
    if min_class_photos == 1 and masks:
        epochs -= 5
        reasons.append("хотя бы одна фация показана только на одном фото")
    if quality.severe_imbalance:
        epochs -= 5
        reasons.append("сильный естественный дисбаланс: эпохи не используются как замена новым данным")

    known = None if known_model_classes is None else {name.strip() for name in known_model_classes if name.strip()}
    new_classes = tuple(sorted((set(class_counts) - known), key=str.casefold)) if known is not None else ()
    if new_classes:
        addition = min(20, 6 * len(new_classes))
        epochs += addition
        reasons.append(f"{len(new_classes)} новых для модели фаций: добавлен запас на обучение головы классов")
    elif known is None:
        reasons.append("у выбранной модели нет каталога: наличие новых классов оценить нельзя")

    suspicious = quality.duplicate_masks + quality.too_small_masks + quality.too_large_masks
    if suspicious:
        epochs -= 5
        reasons.append("есть подозрительные размеры или дубликаты масок: максимум снижен")

    recommended = _round_five(max(15, min(100, epochs)))
    confidence = "высокая"
    if photos < 4 or min_class < 3 or min_class_photos < 2 or suspicious:
        confidence = "низкая"
    elif photos < 10 or min_class < 5 or quality.severe_imbalance or known is None:
        confidence = "средняя"

    # The UI starts training with this ceiling.  Ultralytics may stop earlier
    # after ``patience`` flat validation epochs and keeps its best checkpoint.
    ceiling_margin = 15 if confidence == "низкая" else max(15, _round_five(recommended * 0.35))
    maximum = min(150, recommended + ceiling_margin)
    patience = max(8, min(20, round(recommended * 0.25)))
    return TrainingRecommendation(
        recommended_epochs=recommended,
        max_epochs=maximum,
        patience=patience,
        confidence=confidence,
        verified_masks=masks,
        source_photos=photos,
        wells=wells,
        classes=classes,
        min_class_examples=min_class,
        min_class_photos=min_class_photos,
        new_classes=new_classes,
        reasons=tuple(reasons),
    )


def _round_five(value: float) -> int:
    return int(5 * round(float(value) / 5.0))
