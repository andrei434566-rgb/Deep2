"""Console entry point for Kern Analyzer's no-dialog Excel mask pipeline.

Example:
    python run_kern_analyzer_pipeline.py "D:\\core\\P-31" "D:\\core\\P-31\\P-31.xlsx"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.infrastructure.kern_analyzer_pipeline import KernAnalyzerAutomaticPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Kern Analyzer: automatic standard Excel-to-facies-mask pipeline.")
    parser.add_argument("photo_folder", type=Path, help="Folder containing the core photos")
    parser.add_argument("excel", type=Path, help="Excel file with facies top/base and name")
    parser.add_argument("--output", type=Path, help="New result folder; default is _kern_analyzer_excel_masks in photo folder")
    args = parser.parse_args()
    output = args.output or args.photo_folder / "_kern_analyzer_excel_masks"
    try:
        result = KernAnalyzerAutomaticPipeline().run(args.photo_folder, args.excel, output)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Done. Photos: {result.photos_labeled}/{result.photos_seen}; masks: {result.masks_created}; classes: {len(result.classes)}")
    print(f"Review: {result.output_dir / 'review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
