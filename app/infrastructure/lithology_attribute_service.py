"""Independent visual attribute suggester for already segmented facies."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QPolygonF

from app.domain.models import FaciesDetection, PhotoRecord


class LithologyAttributeService:
    """Suggest observable attributes without changing the facies segmentation.

    Facies boundaries remain the responsibility of YOLO.  This service works
    only inside each accepted mask and deliberately returns suggestions (not a
    geological conclusion) for colour, apparent grain size and bedding.
    """

    def suggest(self, record: PhotoRecord, detection: FaciesDetection) -> dict[str, str]:
        try:
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(Path(record.path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or record.pixmap.isNull():
                return {}
            rect = QPolygonF(detection.polygon).boundingRect()
            x_scale = image.shape[1] / max(1, record.pixmap.width())
            y_scale = image.shape[0] / max(1, record.pixmap.height())
            left = max(0, int(rect.left() * x_scale))
            top = max(0, int(rect.top() * y_scale))
            right = min(image.shape[1], int(rect.right() * x_scale))
            bottom = min(image.shape[0], int(rect.bottom() * y_scale))
            crop = image[top:bottom, left:right]
            if crop.size == 0:
                return {}
            return self._suggest_crop(crop)
        except (ImportError, OSError):
            return {}

    @staticmethod
    def _suggest_crop(crop) -> dict[str, str]:
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = [float(channel.mean()) for channel in cv2.split(hsv)]
        if saturation > 45 and 5 <= hue <= 28:
            colour = "Коричневый" if value < 145 else "Буроватый"
        elif value < 55:
            colour = "Черный"
        elif value < 95:
            colour = "Темно-серый"
        elif value < 165:
            colour = "Серый"
        else:
            colour = "Светло-серый"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        grain = "Тонкозернистый" if texture < 35 else "Мелкозернистый" if texture < 110 else "Среднезернистый" if texture < 260 else "Крупнозернистый"
        gx = float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean())
        gy = float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean())
        bedding = "Субгоризонтальная" if gy > gx * 1.35 else "Волнистая" if abs(gx - gy) / max(1.0, gx + gy) < 0.10 else ""
        values = {"Цвет": colour, "Зернистость": grain}
        if bedding:
            values["Слоистость"] = bedding
        return values
