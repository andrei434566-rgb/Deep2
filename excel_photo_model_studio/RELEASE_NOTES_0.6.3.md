# Excel Photo Model Studio 0.6.3

## Improvements

- Collects CUDA runtime files directly from the installed `nvidia` package tree so the Windows build cannot silently omit wheel DLLs because of package metadata path differences.
- Verifies both the expected CUDA runtime DLL names and their packaged `nvidia` directory before starting the portable-app self-test.
- Retains the interval-conditioned short-description model and the 0.6.1/0.6.2 fixes.

## Build

The Windows x64 portable ZIP includes the application, PyTorch CUDA runtime, and Tesseract OCR. Extract the complete archive before starting the executable.
