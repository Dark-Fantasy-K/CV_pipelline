"""
PersonPoseHandler
─────────────────
检测到 person 后，用 YOLOv8s-pose 做姿态估计，
返回每个人的 17 个 COCO 关键点。
"""

import time
from typing import Dict, List, Any
import numpy as np
import cv2
from ultralytics import YOLO

from pipeline import BaseHandler, Detection

# COCO 17 关键点名称
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 骨架连线 (索引对)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
    (5, 11), (6, 12), (11, 12),                # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),    # 下肢
]

# 配色
COLOR_KP = (110, 231, 183)       # 关键点 - 绿
COLOR_SKEL = (80, 180, 140)      # 骨架线 - 深绿
COLOR_BBOX = (110, 231, 183)     # 边框
COLOR_TEXT_BG = (18, 18, 26)     # 标签底色


class PersonPoseHandler(BaseHandler):
    target_classes = ["person"]

    def init_models(self):
        print("  Loading YOLOv8s-pose...")
        t0 = time.time()
        self.pose_model = YOLO("yolov8s-pose.pt")
        print(f"  YOLOv8s-pose loaded in {time.time() - t0:.2f}s")

    def process(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        对整张图跑一次 pose 推理，然后将关键点匹配回各检测框。
        """
        results = self.pose_model(frame, verbose=False)[0]

        persons = []
        if results.keypoints is not None:
            kpts_data = results.keypoints.data.cpu().numpy()   # (N, 17, 3)
            boxes = results.boxes

            for i in range(len(kpts_data)):
                kpts = kpts_data[i]  # (17, 3)  x, y, conf
                bbox = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])

                keypoints = {}
                for j, name in enumerate(KEYPOINT_NAMES):
                    x, y, c = float(kpts[j][0]), float(kpts[j][1]), float(kpts[j][2])
                    keypoints[name] = {
                        "x": round(x, 1),
                        "y": round(y, 1),
                        "confidence": round(c, 3),
                    }

                persons.append({
                    "bbox": [round(v, 1) for v in bbox],
                    "confidence": round(conf, 4),
                    "keypoints": keypoints,
                })

        return {
            "task": "pose_estimation",
            "person_count": len(persons),
            "persons": persons,
        }

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        process_result: Dict[str, Any],
    ) -> np.ndarray:
        """绘制关键点 + 骨架 + 边框"""
        img = frame.copy()

        for person in process_result.get("persons", []):
            # 边框
            x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)

            # 标签
            label = f'person {person["confidence"]:.0%}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), COLOR_TEXT_BG, -1)
            cv2.putText(img, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BBOX, 1, cv2.LINE_AA)

            # 关键点坐标
            kpts = person["keypoints"]
            pts = []
            for name in KEYPOINT_NAMES:
                kp = kpts[name]
                pts.append((int(kp["x"]), int(kp["y"]), kp["confidence"]))

            # 画骨架
            for i, j in SKELETON:
                if pts[i][2] > 0.3 and pts[j][2] > 0.3:
                    cv2.line(img, (pts[i][0], pts[i][1]),
                             (pts[j][0], pts[j][1]), COLOR_SKEL, 2, cv2.LINE_AA)

            # 画关键点
            for px, py, pc in pts:
                if pc > 0.3:
                    cv2.circle(img, (px, py), 4, COLOR_KP, -1, cv2.LINE_AA)

        return img
