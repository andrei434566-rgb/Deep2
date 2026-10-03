# Excel Photo Model Studio 0.6.7

## Improvements

- Adds a separate YOLO Segmentation export tab. It exports every fully confirmed well from the accumulated cache without starting training, and reports the photo, mask, class, and train/validation counts.
- Exports Ultralytics-compatible polygon labels, `data.yaml`, class metadata, and the approved short-description dataset; refuses to overwrite an existing destination folder.
- Keeps manually corrected photo intervals when OCR is re-read, instead of silently replacing the user's depths with a conflicting OCR range.
- Produces straight rectangular interval masks within the detected core columns, avoiding ragged contours from striations and image noise.
- Adds bulk confirmation controls for reviewed photos and mask rows.

## Validation

- Full application test suite passes: 167 tests.
- The Windows release workflow builds the portable CUDA 12.6 application, verifies the bundled CUDA runtime, and runs the packaged self-test before publishing split ZIP archives and SHA-256 checksums.
