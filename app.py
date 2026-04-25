"""
YOLOv8 Pipeline Server
──────────────────────
YOLO 检测 → Pipeline 分发 → Handler 处理

  person       → PersonPoseHandler  (姿态估计)
  car/truck/…  → VehicleCountHandler (计数跟踪)
  其他          → DefaultHandler     (透传)
"""

import io
import time
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from ultralytics import YOLO

from pipeline import Pipeline, Detection
from handlers import PersonPoseHandler, VehicleCountHandler, DefaultHandler

app = Flask(__name__)

# ---------- 初始化 ----------
print("=" * 50)
print("Initializing YOLOv8 Pipeline Server")
print("=" * 50)

# 1) 主检测器
print("\n[1/2] Loading YOLOv8s detector...")
t0 = time.time()
detector = YOLO("yolov8s.pt")
print(f"  YOLOv8s loaded in {time.time() - t0:.2f}s")

# 2) Pipeline + Handlers
print("\n[2/2] Registering handlers...")
pipeline = Pipeline()
pipeline.register(PersonPoseHandler())
pipeline.register(VehicleCountHandler())
pipeline.set_default(DefaultHandler())

print("\n" + "=" * 50)
print("Server ready!")
print("=" * 50 + "\n")


# ---------- 路由 ----------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "detector": "yolov8s",
        "handlers": {
            "person": "PersonPoseHandler (pose estimation)",
            "car/truck/bus/motorcycle": "VehicleCountHandler (counting & tracking)",
            "default": "DefaultHandler (passthrough)",
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image field in request"}), 400

    start = time.time()

    # 读取图片
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    frame = np.array(img)

    # 1) YOLO 检测
    yolo_results = detector(frame, verbose=False)[0]

    detections = []
    for box in yolo_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(
            class_name=yolo_results.names[int(box.cls)],
            confidence=round(float(box.conf), 4),
            bbox=[round(v, 1) for v in [x1, y1, x2, y2]],
        ))

    # 2) Pipeline 分发处理
    result = pipeline.run(detections, frame)

    # 3) 编码标注图
    annotated_rgb = Image.fromarray(result["annotated_frame"])
    buf = io.BytesIO()
    annotated_rgb.save(buf, format="JPEG", quality=85)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode()

    latency_ms = round((time.time() - start) * 1000, 1)

    # 4) 构造响应
    response = {
        "latency_ms":      latency_ms,
        "total_detections": len(detections),
        "annotated_img":   f"data:image/jpeg;base64,{annotated_b64}",
        "unhandled":       result["unhandled"],
    }

    # 把每个 handler 的结果合并进去
    for handler_name, handler_result in result["handler_results"].items():
        response[handler_name] = handler_result

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
