import cv2
import shutil
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO


def stage_1_main(video_path, output_root):
    yolo_model = YOLO(r".\model\stage1.pt")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{video_path}")
        return {}

    term_idx = 1
    writing = False
    frame_idx = 0
    frame_map = {}
    clarity_map = {}
    current_folder = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = yolo_model.track(frame, verbose=False)[0]
        has_box = len(results.boxes) > 0

        if has_box:
            if not writing:
                current_folder = output_root / f"term{term_idx}"
                current_folder.mkdir(exist_ok=True)
                frame_map[term_idx] = []
                clarity_map[term_idx] = []
                writing = True

            frame_path = current_folder / f"{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            clarity = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            frame_map[term_idx].append(frame_path)
            clarity_map[term_idx].append((frame_path, clarity))
            print(f"📁 儲存 frame 至 term{term_idx}: {frame_path.name}")
        else:
            if writing:
                term_idx += 1
                writing = False

    cap.release()

    term_data = {}
    for term, entries in clarity_map.items():
        if not entries:
            continue
        best_frame_path, _ = max(entries, key=lambda x: x[1])
        image = cv2.imread(str(best_frame_path))
        result = yolo_model(image, verbose=False)[0]
        if result.boxes:
            x1, y1, x2, y2 = map(int, result.boxes[0].xyxy[0])
            patch = image[y1:y2, x1:x2]
            if patch.size > 0:
                term_data[term] = {
                    "frame_path": best_frame_path,
                    "patch": patch
                }

    return term_data