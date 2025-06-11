import cv2
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR


def stage_3_main(classified_data):
    pill_model = YOLO(r".\model\stage3_pill.pt")
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")

    result_images = []
    result_labels = []

    for term, data in classified_data.items():
        patch = data["patch"]
        face_type = data["face_type"]
        display_img = patch.copy()

        if face_type == 0:
            # 文字面：使用 OCR
            ocr_result = ocr.ocr(patch, cls=True)
            found_acetal = False
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    text = line[1][0]
                    if "Acetal" in text:
                        found_acetal = True
                    bbox = line[0]
                    p1 = tuple(map(int, bbox[0]))
                    p2 = tuple(map(int, bbox[2]))
                    cv2.rectangle(display_img, p1, p2, (0, 255, 0), 2)
            result_text = f"AI判斷: {'Acetal' if found_acetal else '其他'}"
        else:
            # 藥丸面：用 pill_model 偵測
            pill_result = pill_model(patch, verbose=False)[0]
            if pill_result.boxes:
                pill_cls = int(pill_result.boxes[0].cls[0])
                pill_name = pill_result.names[pill_cls]
                result_text = f"AI判斷: {pill_name}"
                for sub_box in pill_result.boxes:
                    sx1, sy1, sx2, sy2 = map(int, sub_box.xyxy[0])
                    cv2.rectangle(display_img, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
            else:
                result_text = "AI判斷: 無法辨識"

        result_images.append(display_img)
        result_labels.append((term, result_text))

    return result_images, result_labels
