from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cfg import device
from utils import (
    match_anchors_to_ground_truth,
    encode_boxes,
    descale_bbox,
    draw_bbox_and_save,
    nms,
    decode_boxes,
)
from PIL import Image
import os
from tqdm import tqdm
from torch import nn
from itertools import product as product
from jaxtyping import Num
import cv2
import numpy as np


class SSHContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSHContextModule, self).__init__()
        assert out_channels % 4 == 0, "out_channels should be divisible by 4"

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv1_1_out = self.conv1_1(x)
        conv2_1_out = self.conv2_1(conv1_1_out)
        conv2_2_out = self.conv2_2(conv1_1_out)
        conv3_1_out = self.conv3_1(conv2_2_out)
        output = torch.cat((conv2_1_out, conv3_1_out), dim=1)

        return output


class SSHDetectionModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_anchors=2, kernel_size=3, stride=1
    ):
        super(SSHDetectionModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.context_module = SSHContextModule(in_channels, out_channels)
        self.cls_conv = nn.Conv2d(out_channels * 2, num_anchors * 2, kernel_size=1)
        self.reg_conv = nn.Conv2d(out_channels * 2, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        context_out = self.context_module(x)
        concat_out = torch.cat((conv1_out, context_out), dim=1)
        cls_out = self.cls_conv(concat_out)
        reg_out = self.reg_conv(concat_out)

        return cls_out, reg_out


class SSH(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(SSH, self).__init__()

        self.backbone = backbone
        self.detection_module1 = SSHDetectionModule(128, 128)
        self.detection_module2 = SSHDetectionModule(512, 256)
        self.detection_module3 = SSHDetectionModule(512, 256)

        self.dimension_reduction1 = nn.Sequential(
            nn.Conv2d(
                1280,
                128,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.dimension_reduction2 = nn.Sequential(
            nn.Conv2d(
                96,
                128,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv5_3 = nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1)
        self.mp5_3 = nn.MaxPool2d(2, 2)

        self.conv_end = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, c4, c5 = self.backbone(x)
        conv5_3_out = self.conv5_3(c5)
        mp5_3_out = self.mp5_3(conv5_3_out)

        detection_module3_out = self.detection_module3(mp5_3_out)
        detection_module2_out = self.detection_module2(conv5_3_out)

        dimension_reduction1_out = self.dimension_reduction1(c5)
        dimension_reduction2_out = self.dimension_reduction2(c4)
        upsampling_out = self.upsampling(dimension_reduction1_out)

        eltwise_sum = upsampling_out + dimension_reduction2_out
        conv_end_out = self.conv_end(eltwise_sum)
        detection_module1_out = self.detection_module1(conv_end_out)

        return detection_module1_out, detection_module2_out, detection_module3_out


def train_epoch(
    model: SSH,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    anchors_list: List[Num[torch.Tensor, "1 N 4"]],
    epoch: int,
):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        imgs = batch["image"].to(device)
        gt_boxes = torch.stack(batch["bbox"], dim=0).to(device)

        outputs = model(imgs)

        batch_cls_loss = torch.tensor(0.0, device=device)
        batch_reg_loss = torch.tensor(0.0, device=device)

        for head_idx, (cls_pred, reg_pred) in enumerate(outputs):
            B, _, H, W = cls_pred.shape
            num_anchors = 2

            cls_pred = cls_pred.view(B, num_anchors, 2, H, W)
            cls_pred = cls_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 2)

            reg_pred = reg_pred.view(B, num_anchors, 4, H, W)
            reg_pred = reg_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 4)

            anchors = anchors_list[head_idx]
            anchors = anchors.expand(B, -1, -1)

            if gt_boxes.numel() == 0 or gt_boxes.shape[1] == 0:
                labels = torch.zeros(anchors.shape[:2], dtype=torch.long, device=device)
                matched_gt = torch.zeros_like(anchors)
            else:
                labels, matched_gt = match_anchors_to_ground_truth(anchors, gt_boxes)

            targets = encode_boxes(anchors, matched_gt)

            cls_pred_flat = cls_pred.reshape(-1, 2)
            reg_pred_flat = reg_pred.reshape(-1, 4)
            labels_flat = labels.reshape(-1)
            targets_flat = targets.reshape(-1, 4)

            cls_loss = F.cross_entropy(cls_pred_flat, labels_flat, ignore_index=-1)

            pos_mask = labels_flat == 1
            if pos_mask.sum() > 0:
                reg_loss = F.smooth_l1_loss(
                    reg_pred_flat[pos_mask], targets_flat[pos_mask]
                )
            else:
                reg_loss = torch.tensor(0.0, device=device)

            batch_cls_loss += cls_loss
            batch_reg_loss += reg_loss

        loss = batch_cls_loss + batch_reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += batch_cls_loss.item()
        total_reg_loss += batch_reg_loss.item()
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "cls": f"{batch_cls_loss.item():.4f}",
                "reg": f"{batch_reg_loss.item():.4f}",
            }
        )

    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches

    return avg_loss, avg_cls_loss, avg_reg_loss


def validate(
    model: SSH,
    val_loader: DataLoader,
    anchors_list: List[Num[torch.Tensor, "1 N 4"]],
    epoch: int,
):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_cls_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    num_batches = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            imgs = batch["image"].to(device)
            gt_boxes = torch.stack(batch["bbox"], dim=0).to(device)
            if gt_boxes.numel() == 0 or gt_boxes.shape[1] == 0:
                print(f"Skipping batch {batch_idx} (no ground truth boxes)")
                continue

            outputs = model(imgs)

            batch_cls_loss = torch.tensor(0.0, device=device)
            batch_reg_loss = torch.tensor(0.0, device=device)

            for head_idx, (cls_pred, reg_pred) in enumerate(outputs):
                B, _, H, W = cls_pred.shape
                num_anchors = 2

                cls_pred = cls_pred.view(B, num_anchors, 2, H, W)
                cls_pred = cls_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 2)

                reg_pred = reg_pred.view(B, num_anchors, 4, H, W)
                reg_pred = reg_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 4)

                anchors = anchors_list[head_idx]
                anchors = anchors.expand(B, -1, -1)

                if gt_boxes.numel() == 0 or gt_boxes.shape[1] == 0:
                    labels = torch.zeros(
                        anchors.shape[:2], dtype=torch.long, device=device
                    )
                    matched_gt = torch.zeros_like(anchors)
                else:
                    labels, matched_gt = match_anchors_to_ground_truth(
                        anchors, gt_boxes
                    )
                targets = encode_boxes(anchors, matched_gt)

                cls_pred_flat = cls_pred.reshape(-1, 2)
                reg_pred_flat = reg_pred.reshape(-1, 4)
                labels_flat = labels.reshape(-1)
                targets_flat = targets.reshape(-1, 4)

                cls_loss = F.cross_entropy(cls_pred_flat, labels_flat, ignore_index=-1)

                pos_mask = labels_flat == 1
                if pos_mask.sum() > 0:
                    reg_loss = F.smooth_l1_loss(
                        reg_pred_flat[pos_mask], targets_flat[pos_mask]
                    )
                else:
                    reg_loss = torch.tensor(0.0, device=device)

                batch_cls_loss += cls_loss
                batch_reg_loss += reg_loss

            loss = batch_cls_loss + batch_reg_loss
            total_loss += loss.item()
            total_cls_loss += batch_cls_loss.item()
            total_reg_loss += batch_reg_loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "cls": f"{batch_cls_loss.item():.4f}",
                    "reg": f"{batch_reg_loss.item():.4f}",
                }
            )

    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches

    print(
        f"Validation: Loss: {avg_loss:.4f} | Cls: {avg_cls_loss:.4f} | Reg: {avg_reg_loss:.4f}"
    )

    return avg_loss, avg_cls_loss, avg_reg_loss


def test_and_visualize(
    model: SSH,
    test_loader: DataLoader,
    anchors_list: List[Num[torch.Tensor, "1 N 4"]],
    output_dir: str = "test_outputs",
    n_out: int | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", total=n_out)
        for batch_idx, batch in enumerate(pbar):
            if n_out is not None and batch_idx > n_out:
                break

            imgs = batch["image"].to(device)
            original_images = batch["original_images"]
            bbox = batch.get("bbox")

            outputs = model(imgs)

            all_boxes = []
            all_scores = []

            for head_idx, (cls_pred, reg_pred) in enumerate(outputs):
                B, _, H, W = cls_pred.shape
                num_anchors = 2

                cls_pred = cls_pred.view(B, num_anchors, 2, H, W)
                cls_pred = cls_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 2)

                reg_pred = reg_pred.view(B, num_anchors, 4, H, W)
                reg_pred = reg_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 4)

                anchors = anchors_list[head_idx]
                anchors = anchors.expand(B, -1, -1)

                scores = F.softmax(cls_pred, dim=-1)[:, :, 1]

                pred_boxes = decode_boxes(anchors, reg_pred)

                all_boxes.append(pred_boxes)
                all_scores.append(scores)

            all_boxes = torch.cat(all_boxes, dim=1)
            all_scores = torch.cat(all_scores, dim=1)

            for b in range(B):
                boxes = all_boxes[b]
                scores = all_scores[b]
                original_image = original_images[b]
                orig_size = original_image.size

                score_mask = scores > 0.5
                boxes = boxes[score_mask]
                scores = scores[score_mask]

                output_path = os.path.join(output_dir, f"test_result_{batch_idx}.jpg")
                img_pil = batch["image"][b].cpu()
                img_pil = Image.fromarray(
                    (img_pil.permute(1, 2, 0).numpy() * 255).astype("uint8")
                )
                if len(boxes) == 0:
                    pbar.write(f"No detections for image {batch_idx}")
                    if bbox is not None:
                        bbox_descaled = descale_bbox(
                            bbox[b].numpy().tolist(), orig_size, current_dim=(640, 640)
                        )
                        draw_bbox_and_save(
                            original_image, output_path, None, gt_bbox=bbox_descaled
                        )
                        continue

                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, img_cv)
                    continue

                keep_indices = nms(
                    boxes.unsqueeze(0), scores.unsqueeze(0), iou_threshold=0.3
                )[0]

                final_boxes = boxes[keep_indices].cpu().numpy().tolist()

                final_boxes_descaled = descale_bbox(
                    final_boxes, orig_size, current_dim=(640, 640)
                )

                bbox_descaled = None
                if bbox is not None:
                    bbox_descaled = descale_bbox(
                        bbox[b].numpy().tolist(), orig_size, current_dim=(640, 640)
                    )

                draw_bbox_and_save(
                    original_image,
                    output_path,
                    final_boxes_descaled,
                    gt_bbox=bbox_descaled,
                )

                pbar.write(
                    f"Saved test result {batch_idx} with {len(final_boxes_descaled)} detections"
                )
