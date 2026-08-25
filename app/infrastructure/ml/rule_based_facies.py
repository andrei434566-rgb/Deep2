"""Rule-based detector of persistent visual packages in core photographs.

This is deliberately a transparent heuristic, not a geological classifier.
It detects long, low-saturation core columns and splits them where the visible
texture changes persistently. It never assigns a facies name or probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TextureInterval:
    """A visually distinct package in source-image pixel coordinates."""

    column_index: int
    left: int
    right: int
    top: int
    bottom: int
    polygon: tuple[tuple[int, int], ...]
    evidence: str


class RuleBasedFaciesDetector:
    """Find core columns and persistent visual texture changes.

    The detector only returns visual boundaries. Assigning a geological facies is
    intentionally outside its scope.
    """

    def analyse(self, source_bgr: np.ndarray) -> tuple[np.ndarray, list[TextureInterval]]:
        if source_bgr is None or source_bgr.size == 0:
            raise ValueError("Empty image")
        if source_bgr.ndim != 3 or source_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR color image")

        columns = self._find_core_columns(source_bgr)
        intervals: list[TextureInterval] = []
        for index, (left, top, right, bottom) in enumerate(columns, start=1):
            intervals.extend(self._split_column(source_bgr, index, left, top, right, bottom))
        return self._render(source_bgr, intervals), intervals

    @staticmethod
    def _find_core_columns(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect grey, elongated core objects while rejecting a bright background."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # In standard core-box photography, rocks are usually much less saturated
        # than the brown tray. Projection is more reliable than connected components:
        # black tray edges otherwise connect all columns into one large component.
        candidate = (saturation < 70) & (value < 235)
        column_score = RuleBasedFaciesDetector._smooth(candidate.mean(axis=0).astype(np.float32), 7)
        active_columns = column_score >= 0.48
        min_width = max(10, int(width * 0.018))
        boxes: list[tuple[int, int, int, int]] = []
        for left, right in RuleBasedFaciesDetector._runs(active_columns):
            if right - left < min_width or right - left > width * 0.30:
                continue
            row_score = RuleBasedFaciesDetector._smooth(candidate[:, left:right].mean(axis=1).astype(np.float32), 15)
            row_runs = RuleBasedFaciesDetector._runs(row_score >= 0.24)
            if not row_runs:
                continue
            top, bottom = max(row_runs, key=lambda run: run[1] - run[0])
            if bottom - top < height * 0.25:
                continue
            # Preserve a 1-pixel margin so a natural edge is still visible in output.
            boxes.append((max(0, left - 1), max(0, top - 1), min(width, right + 1), min(height, bottom + 1)))

        if boxes:
            return boxes
        return RuleBasedFaciesDetector._fallback_component_boxes(candidate.astype(np.uint8) * 255)

    @staticmethod
    def _fallback_component_boxes(candidate: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Fallback for photos whose tray and core do not differ in saturation."""
        height, width = candidate.shape
        connected = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(17, int(height * 0.075)))),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        boxes: list[tuple[int, int, int, int]] = []
        for component in range(1, count):
            left, top, box_width, box_height, area = stats[component]
            aspect = box_height / max(box_width, 1)
            if box_width >= max(10, int(width * 0.018)) and box_width <= width * 0.30 and box_height >= height * 0.25 and 2.0 <= aspect <= 35.0 and area >= box_width * box_height * 0.20:
                boxes.append((int(left), int(top), int(left + box_width), int(top + box_height)))
        return sorted(boxes, key=lambda item: item[0])

    @staticmethod
    def _merge_nearby_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Merge split detections of the same vertically continuous core column."""
        merged: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if not merged:
                merged.append(box)
                continue
            left, top, right, bottom = box
            old_left, old_top, old_right, old_bottom = merged[-1]
            horizontal_overlap = min(right, old_right) - max(left, old_left)
            same_column = horizontal_overlap > min(right - left, old_right - old_left) * 0.55
            small_gap = top - old_bottom < 35
            if same_column and small_gap:
                merged[-1] = (min(left, old_left), min(top, old_top), max(right, old_right), max(bottom, old_bottom))
            else:
                merged.append(box)
        return merged

    def _split_column(
        self,
        image: np.ndarray,
        column_index: int,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> list[TextureInterval]:
        height = bottom - top
        width = right - left
        inset = max(1, int(width * 0.08))
        analysis_left = min(right - 2, left + inset)
        analysis_right = max(analysis_left + 2, right - inset)

        roi = image[top:bottom, analysis_left:analysis_right]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if gray.shape[0] < 20 or gray.shape[1] < 3:
            return []

        # 35th percentile is deliberately robust to white labels covering part
        # of a row while keeping genuinely pale sandstone pale.
        row_brightness = np.quantile(gray, 0.35, axis=1)
        vertical_gradient = np.zeros_like(row_brightness)
        vertical_gradient[1:] = np.mean(np.abs(np.diff(gray, axis=0)), axis=1)

        smooth_window = self._odd(max(9, height // 70))
        brightness = self._smooth(row_brightness, smooth_window)
        lamination = self._smooth(vertical_gradient, self._odd(max(7, height // 95)))
        brightness_z = self._robust_scale(brightness)
        lamination_z = self._robust_scale(lamination)

        # A facies boundary needs a sustained texture change, not one dark line.
        change = np.abs(np.diff(brightness_z, prepend=brightness_z[0])) + 0.65 * np.abs(
            np.diff(lamination_z, prepend=lamination_z[0])
        )
        change = self._smooth(change, self._odd(max(5, height // 160)))
        boundaries = self._persistent_boundaries(change, height)
        edges = [0, *boundaries, height]
        boundary_curves = self._boundary_curves(gray, boundaries, analysis_left)

        intervals: list[TextureInterval] = []
        for local_top, local_bottom in zip(edges[:-1], edges[1:]):
            if local_bottom - local_top < max(8, height // 120):
                continue
            mean_brightness = float(np.mean(brightness_z[local_top:local_bottom]))
            mean_lamination = float(np.mean(lamination_z[local_top:local_bottom]))
            evidence = self._texture_evidence(mean_brightness, mean_lamination)
            intervals.append(
                TextureInterval(
                    column_index=column_index,
                    left=left,
                    right=right,
                    top=top + local_top,
                    bottom=top + local_bottom,
                    polygon=self._package_polygon(
                        left,
                        right,
                        analysis_left,
                        analysis_right,
                        top,
                        local_top,
                        local_bottom,
                        boundary_curves,
                    ),
                    evidence=evidence,
                )
            )
        return intervals

    @staticmethod
    def _boundary_curves(
        gray: np.ndarray,
        boundaries: list[int],
        image_x_start: int,
    ) -> dict[int, list[tuple[int, int]]]:
        """Follow the strongest local contact around each global boundary.

        A single horizontal change-point provides the approximate depth of a
        contact. Searching around it independently across the core width and
        smoothing the resulting picks produces a curved contact where the photo
        supports one, while keeping noisy laminae from making a jagged contour.
        """
        height, width = gray.shape
        vertical_edges = np.abs(np.diff(gray, axis=0))
        vertical_edges = cv2.GaussianBlur(vertical_edges, (5, 9), 0)
        curves: dict[int, list[tuple[int, int]]] = {}
        # Limit the local search so a white sample label or a fracture cannot pull
        # the entire contact tens of pixels away from the persistent global change.
        search_radius = max(5, min(14, height // 30))

        for boundary in boundaries:
            values: list[float] = []
            low = max(1, boundary - search_radius)
            high = min(height - 2, boundary + search_radius)
            for x in range(width):
                profile = vertical_edges[low:high + 1, x]
                values.append(float(low + int(np.argmax(profile))))
            smooth_values = RuleBasedFaciesDetector._smooth(np.asarray(values, dtype=np.float32), 17)
            curves[boundary] = [
                (image_x_start + x, int(np.clip(round(y), low, high)))
                for x, y in enumerate(smooth_values)
            ]
        return curves

    @staticmethod
    def _package_polygon(
        left: int,
        right: int,
        analysis_left: int,
        analysis_right: int,
        image_top: int,
        local_top: int,
        local_bottom: int,
        curves: dict[int, list[tuple[int, int]]],
    ) -> tuple[tuple[int, int], ...]:
        width = analysis_right - analysis_left
        if local_top in curves:
            top_curve = [(x, image_top + y) for x, y in curves[local_top]]
        else:
            top_curve = [(analysis_left + x, image_top + local_top) for x in range(width)]
        if local_bottom in curves:
            bottom_curve = [(x, image_top + y) for x, y in curves[local_bottom]]
        else:
            bottom_curve = [(analysis_left + x, image_top + local_bottom) for x in range(width)]

        # Traverse the top contact left-to-right, then the right and bottom sides,
        # then return along the left side. This is a real polygon, not a rectangle.
        points = [
            (left, top_curve[0][1]),
            *top_curve,
            (right - 1, top_curve[-1][1]),
            (right - 1, bottom_curve[-1][1]),
            *reversed(bottom_curve),
            (left, bottom_curve[0][1]),
        ]
        return tuple(points)

    @staticmethod
    def _persistent_boundaries(change: np.ndarray, height: int) -> list[int]:
        # Conservative defaults: the tool should flag a persistent package change,
        # not turn every visible lamina into a separate facies interval.
        minimum_package_height = max(28, int(height * 0.075))
        threshold = max(float(np.quantile(change, 0.94)), float(np.median(change) + 1.85 * np.median(np.abs(change - np.median(change)))))
        candidates = [
            index
            for index in range(2, len(change) - 2)
            if change[index] >= threshold and change[index] >= change[index - 1] and change[index] >= change[index + 1]
        ]

        # Keep strongest boundaries first, then enforce a package-height gap.
        accepted: list[int] = []
        for index in sorted(candidates, key=lambda item: float(change[item]), reverse=True):
            if index < minimum_package_height or height - index < minimum_package_height:
                continue
            if all(abs(index - existing) >= minimum_package_height for existing in accepted):
                accepted.append(index)
        return sorted(accepted)

    @staticmethod
    def _texture_evidence(brightness: float, lamination: float) -> str:
        if lamination >= 0.58:
            return "high lamination density"
        if brightness >= 0.64:
            return "pale comparatively uniform texture"
        if brightness <= 0.35:
            return "dark comparatively uniform texture"
        return "persistent change of visual texture"

    @staticmethod
    def _render(source: np.ndarray, intervals: list[TextureInterval]) -> np.ndarray:
        result = source.copy()
        overlay = result.copy()
        color = (240, 190, 40)  # neutral cyan in BGR; category is not encoded by color
        for interval in intervals:
            cv2.rectangle(overlay, (interval.left, interval.top), (interval.right - 1, interval.bottom - 1), color, -1)
        result = cv2.addWeighted(overlay, 0.16, result, 0.84, 0)

        for interval in intervals:
            cv2.rectangle(result, (interval.left, interval.top), (interval.right - 1, interval.bottom - 1), color, 2)
        return result

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values.copy()
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        return np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode="valid")

    @staticmethod
    def _robust_scale(values: np.ndarray) -> np.ndarray:
        lower, upper = np.quantile(values, [0.05, 0.95])
        if upper - lower < 1e-6:
            return np.full_like(values, 0.5)
        return np.clip((values - lower) / (upper - lower), 0.0, 1.0)

    @staticmethod
    def _odd(value: int) -> int:
        return value if value % 2 else value + 1

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """Return [start, end) intervals of contiguous true values."""
        padded = np.pad(mask.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return list(zip(starts.tolist(), ends.tolist()))
