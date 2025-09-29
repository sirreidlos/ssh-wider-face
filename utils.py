from PIL import Image
from typing import Tuple, List
import cv2
import numpy as np


def get_img_and_bbox(t: dict) -> Tuple[Image.Image, List[List[float]]]:
    return (t["image"], t["faces"]["bbox"])


def pad_to(
    img: Image.Image,
    bbox: List[List[float]],
    target_dim: Tuple[int, int] = (640, 640),
) -> Tuple[Image.Image, List[List[float]]]:
    w, h = img.size
    target_w, target_h = target_dim

    im_ratio = float(h) / w
    model_ratio = float(target_h) / target_w

    if im_ratio > model_ratio:
        new_h = target_h
        new_w = int(new_h / im_ratio)
    else:
        new_w = target_w
        new_h = int(new_w * im_ratio)

    img_np = np.array(img)
    resized = cv2.resize(img_np, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

    det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    det_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    scale_x = new_w / w
    scale_y = new_h / h
    scaled_bboxes = []
    for x, y, bw, bh in bbox:
        new_x = x * scale_x + x_offset
        new_y = y * scale_y + y_offset
        new_bw = bw * scale_x
        new_bh = bh * scale_y
        scaled_bboxes.append([new_x, new_y, new_bw, new_bh])

    # Convert back to PIL
    det_img_pil = Image.fromarray(det_img)

    return det_img_pil, scaled_bboxes


def draw_bbox_and_save(img_pil, bboxes, filename):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    for x, y, w, h in bboxes:
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(img_cv, pt1, pt2, (0, 255, 0), 2)  # green box, thickness=2

    cv2.imwrite(filename, img_cv)
    print("Wrote to", filename)
