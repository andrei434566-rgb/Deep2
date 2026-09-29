# Excel Photo Model Studio 0.6.2

## Improvements

- Keeps the 0.6.1 fixes for narrow partial core columns and normalized confirmed-well cache paths.
- Verifies bundled CUDA DLLs across the complete PyInstaller onedir layout rather than assuming a fixed internal directory.
- Separates CUDA-runtime verification from the packaged-app self-test so failures identify the exact stage.
- Retains the interval-conditioned short-description model, trained from reviewed core crops and Excel descriptions.

## Build

The Windows x64 portable ZIP includes the application, PyTorch CUDA runtime, and Tesseract OCR. Extract the complete archive before starting the executable.
