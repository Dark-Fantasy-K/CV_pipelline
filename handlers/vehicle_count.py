"""
VehicleCountHandler
───────────────────
检测到 car / truck / bus / motorcycle 后，
进行计数统计 + 简易 IoU 跟踪（跨帧可用）。
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import cv2

from pipeline import BaseHandler, Detection

# 配色方案（按类别）
CLASS_COLORS = {
    "car":        (251, 146, 60),   # 橙
    "truck":      (96, 165, 250),   # 蓝
    "bus":        (250, 204, 21),   # 黄
    "motorcycle": (192, 132, 252),  # 紫
}
DEFAULT_COLOR = (200, 200, 200)
COLOR_TEXT_BG = (18, 18, 26)


def _iou(box_a, box_b):
    """计算两个 bbox 的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


class SimpleTracker:
    """
    基于 IoU 的简易多目标跟踪器。
    适用于逐帧调用的场景（视频流 / 连续请求）。
    """

    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks: Dict[int, Dict] = {}   # track_id → {bbox, class, lost}
        self.next_id = 1
        self.total_count: Dict[str, int] = defaultdict(int)  # 累计经过的车辆

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        用当前帧的检测更新跟踪状态，返回带 track_id 的检测列表。
        """
        updated_dets = []
        used_tracks = set()

        # 贪心匹配：对每个检测找最大 IoU 的已有 track
        for det in sorted(detections, key=lambda d: -d.confidence):
            best_id, best_iou = None, 0
            for tid, trk in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou(det.bbox, trk["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None and best_iou >= self.iou_threshold:
                # 匹配成功 → 更新 track
                self.tracks[best_id]["bbox"] = det.bbox
                self.tracks[best_id]["lost"] = 0
                det.track_id = best_id
                used_tracks.add(best_id)
            else:
                # 新目标
                det.track_id = self.next_id
                self.tracks[self.next_id] = {
                    "bbox": det.bbox,
                    "class": det.class_name,
                    "lost": 0,
                }
                self.total_count[det.class_name] += 1
                self.next_id += 1

            updated_dets.append(det)

        # 未匹配的 track → lost + 1，超时则删除
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return updated_dets


class VehicleCountHandler(BaseHandler):
    target_classes = ["car", "truck", "bus", "motorcycle"]

    def init_models(self):
        """车辆计数/跟踪不需要额外模型"""
        self.tracker = SimpleTracker(iou_threshold=0.3, max_lost=30)
        print("  Vehicle tracker initialized")

    def process(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        # 更新跟踪器
        tracked = self.tracker.update(detections)

        # 当前帧各类计数
        current_count: Dict[str, int] = defaultdict(int)
        vehicles = []
        for det in tracked:
            current_count[det.class_name] += 1
            vehicles.append({
                "class":      det.class_name,
                "confidence": det.confidence,
                "bbox":       det.bbox,
                "track_id":   det.track_id,
            })

        return {
            "task":          "vehicle_counting",
            "current_frame": dict(current_count),
            "current_total": sum(current_count.values()),
            "cumulative":    dict(self.tracker.total_count),
            "active_tracks": len(self.tracker.tracks),
            "vehicles":      vehicles,
        }

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        process_result: Dict[str, Any],
    ) -> np.ndarray:
        """绘制车辆边框 + track ID + 类别 + 计数面板"""
        img = frame.copy()

        for v in process_result.get("vehicles", []):
            color = CLASS_COLORS.get(v["class"], DEFAULT_COLOR)
            x1, y1, x2, y2 = [int(c) for c in v["bbox"]]

            # 边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 标签
            label = f'#{v["track_id"]} {v["class"]} {v["confidence"]:.0%}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), COLOR_TEXT_BG, -1)
            cv2.putText(img, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 左上角计数面板
        panel_lines = [f"Vehicles: {process_result['current_total']}"]
        for cls_name, cnt in sorted(process_result["current_frame"].items()):
            panel_lines.append(f"  {cls_name}: {cnt}")
        panel_lines.append(f"Tracks: {process_result['active_tracks']}")

        y_offset = 30
        for line in panel_lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (10, y_offset - th - 4), (20 + tw, y_offset + 4),
                          COLOR_TEXT_BG, -1)
            cv2.putText(img, line, (14, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (251, 146, 60), 1, cv2.LINE_AA)
            y_offset += th + 12

        return img
