import matplotlib.pyplot as plt
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np


def draw_text_with_font(img_cv2, text, position=(10, 100), font_size=128, color=(255, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "C:/Windows/Fonts/msjh.ttc"
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_results(result_images, result_labels, save_path):
    num = len(result_images)
    cols = min(3, num)
    rows = (num + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    if rows == 1:
        axs = [axs] if cols == 1 else axs
    axs = axs.flatten()

    for i, (img, (term, label)) in enumerate(zip(result_images, result_labels)):
        annotated_img = draw_text_with_font(img.copy(), label, position=(10, 100), font_size=128, color=(255, 0, 0))
        rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(rgb_img)
        axs[i].set_title(term)
        axs[i].axis("off")

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()