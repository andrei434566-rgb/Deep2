# Excel Photo Model Studio 0.6.6

## Improvements

- Adds a persistent, visible list of fully reviewed wells, with photo, mask, and facies counts. New wells accumulate in the local training catalog without replacing prior wells.
- Adds click-to-select removal and undo for false-positive core-column masks in the step-by-step review.
- Reduces false core-column detections from colored and black down arrows while retaining short core pieces.
- Keeps training CUDA-only and builds the portable application with the CUDA 12.6 PyTorch runtime through the Windows release workflow.
