# Excel Photo Model Studio 0.6.1

## Improvements

- Retains dense narrow partial-core columns while filtering rulers and arrows.
- Normalizes cached photo paths so confirmed wells remain available on Windows installations using short user-folder paths.
- Fixes dataset, text-model, and inference tests to exercise the real interval-description flow.
- Keeps the interval-conditioned short-description model separate from YOLO's standard `best.pt` segmentation checkpoint.

## Build

The Windows x64 portable ZIP includes the application, PyTorch CUDA runtime, and Tesseract OCR. Extract the complete archive before starting the executable.
