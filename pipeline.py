"""
Pipeline 调度器
──────────────
YOLO 检测到的每个目标，根据 class name 分发到对应的 Handler。
支持动态注册 handler，未注册的类别走 default handler。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from PIL import Image


@dataclass
class Detection:
    """单个检测目标的数据结构"""
    class_name: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2]
    track_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseHandler:
    """所有 handler 的基类"""

    # 该 handler 负责处理的 YOLO 类名列表
    target_classes: List[str] = []

    def init_models(self):
        """加载该 handler 专用的模型，启动时调用一次"""
        pass

    def process(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        处理一批同类检测结果。

        Args:
            detections: 属于本 handler 的检测列表
            frame: 原始图像 (RGB numpy array)

        Returns:
            该 handler 的处理结果 dict
        """
        raise NotImplementedError

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        process_result: Dict[str, Any],
    ) -> np.ndarray:
        """
        在图上绘制该 handler 的可视化结果，返回标注后的图像。
        默认不做任何绘制。
        """
        return frame


class Pipeline:
    """Pipeline 主控：注册 handler → 检测 → 分发 → 汇总"""

    def __init__(self):
        self._handlers: Dict[str, BaseHandler] = {}
        self._default_handler: Optional[BaseHandler] = None

    def register(self, handler: BaseHandler):
        """注册一个 handler，自动绑定其 target_classes"""
        handler.init_models()
        for cls_name in handler.target_classes:
            self._handlers[cls_name] = handler
        print(f"  ✓ Registered {handler.__class__.__name__} "
              f"for {handler.target_classes}")

    def set_default(self, handler: BaseHandler):
        """设置兜底 handler（可选）"""
        handler.init_models()
        self._default_handler = handler
        print(f"  ✓ Default handler: {handler.__class__.__name__}")

    def get_handler(self, class_name: str) -> Optional[BaseHandler]:
        return self._handlers.get(class_name, self._default_handler)

    def run(
        self,
        detections: List[Detection],
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        执行完整 pipeline：
        1. 按 handler 分组检测结果
        2. 每组调用对应 handler.process()
        3. 每组调用对应 handler.annotate()
        4. 汇总返回

        Returns:
            {
                "handler_results": { "PersonPoseHandler": {...}, ... },
                "annotated_frame": np.ndarray,
                "unhandled": [ ... ],
            }
        """
        # ---- 1) 按 handler 实例分组 ----
        groups: Dict[BaseHandler, List[Detection]] = {}
        unhandled: List[Detection] = []

        for det in detections:
            handler = self.get_handler(det.class_name)
            if handler is None:
                unhandled.append(det)
            else:
                groups.setdefault(handler, []).append(det)

        # ---- 2) 逐组 process ----
        handler_results: Dict[str, Any] = {}
        annotated = frame.copy()

        for handler, dets in groups.items():
            name = handler.__class__.__name__
            result = handler.process(dets, frame)
            handler_results[name] = result

            # ---- 3) 逐组 annotate ----
            annotated = handler.annotate(annotated, dets, result)

        return {
            "handler_results": handler_results,
            "annotated_frame": annotated,
            "unhandled": [
                {"class": d.class_name, "confidence": d.confidence,
                 "bbox": d.bbox}
                for d in unhandled
            ],
        }
