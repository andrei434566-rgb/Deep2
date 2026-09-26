"""Opt-in real one-epoch pipeline smoke on explicitly synthetic core fixtures.

This checks serialization/runtime compatibility, NOT geological model quality.
Run with the project Python; all generated files stay under outputs/model_audit_*.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "excel_photo_model_studio" / "src"))


def main() -> None:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from excel_photo_model_studio.training import train_bundle
    from excel_photo_model_studio.description_model import DescriptionGenerator
    from app.infrastructure.ml.yolo_model_service import YoloModelService

    audit = ROOT / "outputs" / datetime.now().strftime("model_audit_%Y%m%d_%H%M%S")
    audit.mkdir(parents=True, exist_ok=False)
    dataset = audit / "synthetic_dataset"
    captions = []
    for index in range(6):
        split = "train" if index < 5 else "val"
        for kind in ("images", "labels", "crops"):
            (dataset / kind / split).mkdir(parents=True, exist_ok=True)
        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        rng = np.random.default_rng(index)
        image[8:120, 30:98] = np.clip(130 + rng.normal(0, 15, (112, 68, 3)), 0, 255).astype(np.uint8)
        crop = image[8:120, 30:98]
        image_path = dataset / "images" / split / f"synthetic_{index}.jpg"
        crop_path = dataset / "crops" / split / f"synthetic_{index}.jpg"
        image_path.write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
        crop_path.write_bytes(cv2.imencode(".jpg", crop)[1].tobytes())
        polygon = "0 0.234375 0.0625 0.765625 0.0625 0.765625 0.9375 0.234375 0.9375\n"
        (dataset / "labels" / split / f"synthetic_{index}.txt").write_text(polygon, encoding="utf-8")
        captions.append({
            "split": split, "crop": crop_path.relative_to(dataset).as_posix(),
            "facies": "SyntheticFacies", "target_text": "Synthetic gray core.",
            "source_photo": str(image_path), "well": "SYNTHETIC-NOT-GEOLOGICAL-DATA",
        })
    (dataset / "caption_dataset.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in captions), encoding="utf-8",
    )
    (dataset / "data.yaml").write_text(
        f"path: {json.dumps(dataset.as_posix())}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['SyntheticFacies']\n",
        encoding="utf-8",
    )
    (dataset / "dataset_manifest.json").write_text(json.dumps({
        "schema": "excel-photo-yolo-seg-v1", "class_names": ["SyntheticFacies"],
        "train_caption_count": 5, "val_caption_count": 1,
    }), encoding="utf-8")
    summary = {"synthetic_only": True, "quality_evaluation": False, "output": str(audit)}
    try:
        bundle = train_bundle(
            dataset, audit / "model", epochs=1, patience=1, image_size=64, device="cpu",
            description_epochs=1, description_patience=1,
        )
        best = audit / "model" / "best.pt"
        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        visual = YOLO(str(best))
        results = visual.predict(str(dataset / "images" / "val" / "synthetic_5.jpg"), imgsz=64, device="cpu", verbose=False)
        generator = DescriptionGenerator(best)
        text = generator.generate(dataset / "crops" / "val" / "synthetic_5.jpg", "SyntheticFacies")
        service = YoloModelService(best)
        summary.update({
            "passed": True, "visual_task": visual.task, "visual_results": len(results),
            "embedded_description_schema": checkpoint["core_description_checkpoint"]["schema"],
            "main_analyzer_text_loaded": service._description_generator is not None,
            "generated_characters": len(text), "text_is_untrained_draft": True,
            "bundle": bundle,
        })
        assert visual.task == "segment" and len(results) == 1
        assert service._description_generator is not None
    except Exception as exc:
        summary.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        (audit / "audit_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"AUDIT_RESULT={audit / 'audit_result.json'}", flush=True)


if __name__ == "__main__":
    main()
