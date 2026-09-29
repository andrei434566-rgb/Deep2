# Excel Photo Model Studio 0.6.4

## Improvements

- Adds the PyInstaller internal `torch/lib` folder to Windows' DLL search path as well as CUDA-wheel directories.
- Verifies actual CUDA library names regardless of whether the wheel places them under `nvidia` or `torch/lib`, and reports the discovered DLL names if validation fails.
- Retains the interval-conditioned short-description model and the 0.6.1–0.6.3 fixes.

## Build

The Windows x64 portable ZIP includes the application, PyTorch CUDA runtime, and Tesseract OCR. Extract the complete archive before starting the executable.
