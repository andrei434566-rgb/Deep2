# Excel Photo Model Studio 0.6.0

This release trains and packages two cooperating models for core-photo work:

- `best.pt` is the standard Ultralytics YOLO11-seg model for core/facies masks and facies classes.
- `description_best.pt` is a separate CUDA-trained character decoder. It learns from approved Excel descriptions paired with the pixels inside each reviewed interval mask, the facies index and name, and the complete interval thickness.
- The text model is trained only when there are at least 20 training intervals and 3 independent validation intervals. Otherwise the result includes `description_training_info.json` with the reason and analysis uses the confirmed description example for that facies.
- Analysis applies the generated text to the corresponding predicted interval crop. Low-confidence/invalid output falls back to the confirmed Excel description; neural descriptions remain drafts and must be reviewed.
- YOLO and text-model checkpoints remain separate so `best.pt` stays directly usable by Ultralytics.

The dataset now contains `caption_dataset.jsonl` and mask-only interval crops split with the same well-aware train/validation boundary as YOLO. Pixels outside the reviewed polygon are removed from text-training crops to reduce shortcuts from ruler marks, arrows, and page background.

The release also carries the reviewed-well cache, CUDA 12.6 PyTorch runtime, and the depth/column/mask fixes accumulated since 0.4.8. The bottom-edge centimetre is preserved during interval projection and coverage checks.

`best.pt` and generated descriptions are model candidates, not a guarantee of geological correctness. Review interval boundaries, facies classes, and every generated description on independent wells before relying on them.
