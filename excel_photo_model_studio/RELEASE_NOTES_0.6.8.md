# Excel Photo Model Studio 0.6.8

## Improvements

- Saves individually reviewed masks to the cumulative local cache as soon as at least one valid interval is confirmed; unreviewed masks and errors elsewhere in the project no longer block that save.
- Builds and exports a YOLO segmentation dataset from the confirmed intervals, including a single photo, while clearly marking that an independent validation set is still required before training.
- Keeps earlier immutable snapshots when more masks from the same well are confirmed later.
- Updates the cache catalogue and interface to count only the approved photos and annotations actually included in the dataset.

## Validation

- Full application test suite passes: 168 tests.
- The Windows release workflow builds the portable CUDA 12.6 application, verifies the bundled CUDA runtime, and runs the packaged self-test before publishing split ZIP archives and SHA-256 checksums.
