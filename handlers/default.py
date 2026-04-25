"""
DefaultHandler
──────────────
兜底 handler：对未注册类别仅返回检测信息，不做额外处理。
方便后续扩展新类别时有基础输出。
"""

from typing import Dict, List, Any
import numpy as np
import cv2

from pipeline import BaseHandler, Detection

COLOR_DEFAULT = (160, 160, 180)
COLOR_TEXT_BG = (18, 18, 26)


class DefaultHandler(BaseHandler):
    target_classes = []  # 不绑定具体类，作为 default

    def process(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "task": "passthrough",
            "count": len(detections),
            "classes": list({d.class_name for d in detections}),
            "items": [
                {"class": d.class_name, "confidence": d.confidence, "bbox": d.bbox}
                for d in detections
            ],
        }

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        process_result: Dict[str, Any],
    ) -> np.ndarray:
        img = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_DEFAULT, 1)
            label = f'{det.class_name} {det.confidence:.0%}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), COLOR_TEXT_BG, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DEFAULT, 1, cv2.LINE_AA)
        return img
