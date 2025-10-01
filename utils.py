from PIL import Image
from typing import Tuple, List
from cfg import device
import cv2
import numpy as np
import torch
from jaxtyping import Num


def get_img_and_bbox(t: dict) -> Tuple[Image.Image, List[List[float]]]:
    return (t["image"], t["faces"]["bbox"])


def get_scale_tuple(
    src_dim: Tuple[int, int], dst_dim: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    src_w, src_h = src_dim
    dst_w, dst_h = dst_dim

    src_ratio = float(src_h) / src_w
    dst_ratio = float(dst_h) / dst_w

    if src_ratio > dst_ratio:
        new_h = dst_h
        new_w = int(new_h / src_ratio)
    else:
        new_w = dst_w
        new_h = int(new_w * src_ratio)

    y_offset = (dst_h - new_h) // 2
    x_offset = (dst_w - new_w) // 2

    scale_x = new_w / src_w
    scale_y = new_h / src_h

    return x_offset, y_offset, scale_x, scale_y


def pad_to(
    img: Image.Image,
    target_dim: Tuple[int, int] = (640, 640),
) -> Image.Image:
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

    det_img_pil = Image.fromarray(det_img)

    return det_img_pil


def unpad_from(padded_img: Image.Image, orig_size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = padded_img.size
    orig_w, orig_h = orig_size

    im_ratio = orig_h / orig_w
    model_ratio = target_h / target_w

    if im_ratio > model_ratio:
        new_h = target_h
        new_w = int(new_h / im_ratio)
    else:
        new_w = target_w
        new_h = int(new_w * im_ratio)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    img_np = np.array(padded_img)

    content = img_np[y_offset : y_offset + new_h, x_offset : x_offset + new_w]

    restored = cv2.resize(
        content, dsize=(orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )

    return Image.fromarray(restored)


def scale_bbox(
    bbox: List[List[float]],
    original_dim: Tuple[int, int],
    target_dim: Tuple[int, int] = (640, 640),
) -> List[List[float]]:
    x_offset, y_offset, scale_x, scale_y = get_scale_tuple(original_dim, target_dim)

    scaled_bboxes = []
    for x, y, bw, bh in bbox:
        new_x = x * scale_x + x_offset
        new_y = y * scale_y + y_offset
        new_bw = bw * scale_x
        new_bh = bh * scale_y
        scaled_bboxes.append([new_x, new_y, new_bw, new_bh])

    return scaled_bboxes


def descale_bbox(
    bbox: List[List[float]],
    original_dim: Tuple[int, int],
    current_dim: Tuple[int, int] = (640, 640),
) -> List[List[float]]:
    # print("[DEBUG] bbox", bbox)
    x_offset, y_offset, scale_x, scale_y = get_scale_tuple(original_dim, current_dim)

    scaled_bboxes = []
    for x, y, bw, bh in bbox:
        new_x = round((x - x_offset) / scale_x)
        new_y = round((y - y_offset) / scale_y)
        new_bw = round(bw / scale_x)
        new_bh = round(bh / scale_y)
        scaled_bboxes.append([new_x, new_y, new_bw, new_bh])

    return scaled_bboxes


def draw_bbox_and_save(
    img_pil: Image.Image,
    filename: str,
    pred_bbox: List[List[float]] | None,
    gt_bbox: List[List[float]] | None = None,
):
    # print("[DEBUG] gt_bbox", gt_bbox)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    for x, y, w, h in gt_bbox or []:
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(img_cv, pt1, pt2, (0, 0, 255), 2)

    for x, y, w, h in pred_bbox or []:
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(img_cv, pt1, pt2, (0, 255, 0), 3)

    cv2.imwrite(filename, img_cv)
    print("Wrote to", filename)


def generate_anchors(
    feature_map_size: Tuple[int, int],
    stride: int,
    scales: List[int],
) -> Num[torch.Tensor, "1 N 4"]:
    H, W = feature_map_size
    anchors = []

    for y, x in torch.cartesian_prod(torch.arange(H), torch.arange(W)):
        cy = (y + 0.5) * stride
        cx = (x + 0.5) * stride
        for s in scales:
            anchors.append([cx, cy, s, s])  # center_x, center_y, w, h

    return torch.tensor(anchors, dtype=torch.float32, device=device).unsqueeze(0)


def intersection_over_union(
    boxes1: Num[torch.Tensor, "B N 4"], boxes2: Num[torch.Tensor, "B M 4"]
) -> Num[torch.Tensor, "B N M"]:
    B = boxes1.size(0)

    ious = []

    for b in range(B):
        b1 = boxes1[b]
        b2 = boxes2[b]

        # this one is basically used to find the largest overlap
        b1_x1 = b1[:, 0] - b1[:, 2] / 2
        b1_y1 = b1[:, 1] - b1[:, 3] / 2
        b1_x2 = b1[:, 0] + b1[:, 2] / 2
        b1_y2 = b1[:, 1] + b1[:, 3] / 2

        b2_x1 = b2[:, 0] - b2[:, 2] / 2
        b2_y1 = b2[:, 1] - b2[:, 3] / 2
        b2_x2 = b2[:, 0] + b2[:, 2] / 2
        b2_y2 = b2[:, 1] + b2[:, 3] / 2

        inter_x1 = torch.max(b1_x1[:, None], b2_x1[None])
        inter_y1 = torch.max(b1_y1[:, None], b2_y1[None])
        inter_x2 = torch.min(b1_x2[:, None], b2_x2[None])
        inter_y2 = torch.min(b1_y2[:, None], b2_y2[None])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(
            min=0
        )
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        ious.append(inter_area / (area1[:, None] + area2[None] - inter_area + 1e-6))

    return torch.stack(ious, dim=0)


def match_anchors_to_ground_truth(
    anchors: Num[torch.Tensor, "B N 4"],
    gt_boxes: Num[torch.Tensor, "B M 4"],
    pos_iou: float = 0.5,
    neg_iou: float = 0.3,
) -> Tuple[Num[torch.Tensor, "B N"], Num[torch.Tensor, "B N 4"]]:
    B, N, _ = anchors.shape
    labels_list = []
    matched_list = []

    ious = intersection_over_union(anchors, gt_boxes)
    for b in range(B):
        batch_ious = ious[b]
        max_iou, gt_idx = batch_ious.max(dim=1)

        labels = -1 * torch.ones(N, dtype=torch.long, device=anchors.device)
        labels[max_iou < neg_iou] = 0
        labels[max_iou >= pos_iou] = 1

        matched_gt_boxes = gt_boxes[b][gt_idx]
        labels_list.append(labels)
        matched_list.append(matched_gt_boxes)

    return torch.stack(labels_list, dim=0), torch.stack(matched_list, dim=0)


def encode_boxes(
    anchors: Num[torch.Tensor, "B N 4"],
    gt_boxes: Num[torch.Tensor, "B N 4"],
) -> Num[torch.Tensor, "B N 4"]:
    t_x = (gt_boxes[:, :, 0] - anchors[:, :, 0]) / anchors[:, :, 2]
    t_y = (gt_boxes[:, :, 1] - anchors[:, :, 1]) / anchors[:, :, 3]
    t_w = torch.log(gt_boxes[:, :, 2] / anchors[:, :, 2] + 1e-6)
    t_h = torch.log(gt_boxes[:, :, 3] / anchors[:, :, 3] + 1e-6)
    return torch.stack([t_x, t_y, t_w, t_h], dim=-1)


def decode_boxes(
    anchors: Num[torch.Tensor, "B N 4"],
    deltas: Num[torch.Tensor, "B N 4"],
) -> Num[torch.Tensor, "B N 4"]:
    pred_x = deltas[:, :, 0] * anchors[:, :, 2] + anchors[:, :, 0]
    pred_y = deltas[:, :, 1] * anchors[:, :, 3] + anchors[:, :, 1]
    pred_w = torch.exp(deltas[:, :, 2]) * anchors[:, :, 2]
    pred_h = torch.exp(deltas[:, :, 3]) * anchors[:, :, 3]
    return torch.stack([pred_x, pred_y, pred_w, pred_h], dim=-1)


def nms(
    boxes: Num[torch.Tensor, "B N 4"],
    scores: Num[torch.Tensor, "B N"],
    iou_threshold: float = 0.5,
):
    B = boxes.size(0)
    keep_list = []
    for b in range(B):
        b_boxes = boxes[b]
        b_scores = scores[b]

        x1 = b_boxes[:, 0] - b_boxes[:, 2] / 2
        y1 = b_boxes[:, 1] - b_boxes[:, 3] / 2
        x2 = b_boxes[:, 0] + b_boxes[:, 2] / 2
        y2 = b_boxes[:, 1] + b_boxes[:, 3] / 2

        keep = []
        _, idxs = b_scores.sort(descending=True)

        while idxs.numel() > 0:
            i = idxs[0]
            keep.append(i.item())
            if idxs.numel() == 1:
                break
            rest = idxs[1:]

            xx1 = torch.max(x1[i], x1[rest])
            yy1 = torch.max(y1[i], y1[rest])
            xx2 = torch.min(x2[i], x2[rest])
            yy2 = torch.min(y2[i], y2[rest])

            inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
            iou_val = inter / (
                (x2[i] - x1[i]) * (y2[i] - y1[i])
                + (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])
                - inter
            )
            idxs = rest[iou_val <= iou_threshold]

        keep_list.append(keep)
    return keep_list
