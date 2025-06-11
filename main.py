import os
import json
import cv2
import threading
from pathlib import Path
from stage_1 import stage_1_main
from stage_2 import stage_2_main
from stage_3 import stage_3_main
from draw_result import draw_results


def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["video_path"], config["output_root"]
"""
{
  "video_path": "input video path",
  "output_root": "./result"
}

"""

def play_video(video_path):
    cap_play = cv2.VideoCapture(video_path)
    fps = cap_play.get(cv2.CAP_PROP_FPS)
    while cap_play.isOpened():
        ret, frame = cap_play.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (960, 720))
        cv2.imshow("原始影片播放中", resized_frame)
        if cv2.waitKey(int(1000 // fps)) & 0xFF != 255:
            break
    cap_play.release()
    cv2.destroyAllWindows()


def main():
    video_path, output_root_base = load_config()
    output_root_base = Path(output_root_base)
    output_root_base.mkdir(parents=True, exist_ok=True)

    try_idx = 1
    while (output_root_base / f"try{try_idx}").exists():
        try_idx += 1
    current_try = output_root_base / f"try{try_idx}"
    current_try.mkdir()

    threading.Thread(target=play_video, args=(video_path,), daemon=True).start()

    print("🚀 Stage 1: 擷取最佳幀...")
    term_data = stage_1_main(video_path, current_try)

    print("🚀 Stage 2: 分類文字面 / 藥丸面...")
    classified_data = stage_2_main(term_data)

    print("🚀 Stage 3: OCR 或 藥丸辨識...")
    result_images, result_labels = stage_3_main(classified_data)

    print("🖼️ 繪圖與儲存...")
    draw_results(result_images, result_labels, current_try / "summary.png")


if __name__ == "__main__":
    main()