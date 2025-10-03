import logging
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cfg import device
from torchvision.ops import nms

from utils import (
    match_anchors_to_ground_truth,
    encode_boxes,
    decode_boxes,
)
import os
from tqdm.auto import tqdm
from torch import nn
from itertools import product as product
from jaxtyping import Num
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SSH")


class SSHContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSHContextModule, self).__init__()
        assert out_channels % 4 == 0, "out_channels should be divisible by 4"
        self.logger = logging.getLogger("SSH.ContextModule")
        self.logger.debug(
            f"Initializing with in_channels={in_channels}, out_channels={out_channels}"
        )

        # First branch (3x3 conv with dilation=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.logger.debug(
            f"Branch 1: 3x3 conv (dilation=1) -> {out_channels // 2} channels"
        )

        # Second branch (3x3 conv with dilation=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 4,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.logger.debug(
            f"Branch 2: 3x3 conv (dilation=2) -> {out_channels // 4} channels"
        )

        # Third branch (3x3 conv with dilation=3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 4,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.logger.debug(
            f"Branch 3: 3x3 conv (dilation=3) -> {out_channels // 4} channels"
        )

    def forward(self, x):
        self.logger.debug(f"Input shape: {x.shape}")

        # Process each branch in parallel
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        self.logger.debug(
            f"Branch outputs - b1: {b1.shape}, b2: {b2.shape}, b3: {b3.shape}"
        )

        # Concatenate all branch outputs along channel dimension
        out = torch.cat([b1, b2, b3], dim=1)
        self.logger.debug(f"Output shape: {out.shape}")

        return out


class SSHDetectionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_anchors: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        dropout_rate: float = 0.2,
    ):
        super(SSHDetectionModule, self).__init__()
        self.logger = logging.getLogger("SSH.DetectionModule")
        self.logger.debug(
            f"Initializing with in_channels={in_channels}, out_channels={out_channels}"
        )

        # First conv branch - reduces channels to out_channels//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,  # Half channels for first branch
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,  # Same padding
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

        # Context module branch - processes input and outputs out_channels//2 channels
        self.context_module = SSHContextModule(in_channels, out_channels // 2)
        
        # Dropout layer after context module
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # Output layers with proper initialization
        # Classification: 2 scores per anchor (background/face)
        self.cls_conv = nn.Conv2d(out_channels, num_anchors * 2, kernel_size=1)
        # Regression: 4 coordinates per anchor (dx, dy, dw, dh)
        self.reg_conv = nn.Conv2d(out_channels, num_anchors * 4, kernel_size=1)

        # Initialize weights using Kaiming initialization for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for final layers
        nn.init.normal_(self.cls_conv.weight, std=0.01)
        nn.init.constant_(self.cls_conv.bias, -np.log(0.99/0.01))  # Bias for better initial predictions
        nn.init.normal_(self.reg_conv.weight, std=0.01)
        nn.init.constant_(self.reg_conv.bias, 0)

    def forward(self, x):
        self.logger.debug("-" * 40)
        self.logger.debug(f"Input shape: {x.shape}")
        self.logger.debug(
            f"Module config: in_channels={self.conv1[0].in_channels}, "
            f"out_channels={self.conv1[0].out_channels * 2}, num_anchors=2"
        )

        # First branch: 3x3 conv with dropout
        conv1_out = self.conv1(x)
        self.logger.debug(
            f"After conv1: {conv1_out.shape} "
            f"(should be [B, {self.conv1[0].out_channels}, H, W])"
        )

        # Second branch: Context module
        context_out = self.context_module(x)
        context_out = self.dropout(context_out)  # Apply dropout to context features
        self.logger.debug(
            f"After context_module: {context_out.shape} "
            f"(should be [B, {self.conv1[0].out_channels}, H, W])"
        )

        # Concatenate features from both branches
        concat_out = torch.cat((conv1_out, context_out), dim=1)
        concat_out = F.relu(concat_out)  # Add non-linearity after concat
        self.logger.debug(
            f"After concat: {concat_out.shape} "
            f"(should be [B, {self.conv1[0].out_channels * 2}, H, W])"
        )

        # Classification and regression outputs
        cls_out = self.cls_conv(concat_out)
        reg_out = self.reg_conv(concat_out)

        self.logger.debug("Output shapes:")
        self.logger.debug(
            f"  - Classification: {cls_out.shape} (should be [B, {self.cls_conv.out_channels}, H, W])"
        )
        self.logger.debug(
            f"  - Regression: {reg_out.shape} (should be [B, {self.reg_conv.out_channels}, H, W])"
        )

        return cls_out, reg_out


class SSH(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(SSH, self).__init__()
        self.backbone = backbone
        self.logger = logging.getLogger("SSH.Model")

        # Get output channels from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            out1, out2, out3 = self.backbone(dummy_input)
            self.feat1_channels = out1.shape[1]
            self.feat2_channels = out2.shape[1]
            self.feat3_channels = out3.shape[1]

        self.logger.debug(
            f"Backbone output channels - feat1: {self.feat1_channels}, "
            f"feat2: {self.feat2_channels}, feat3: {self.feat3_channels}"
        )

        # Dimension reduction layers for each feature map (all to 256 channels)
        self.reduce1 = nn.Sequential(
            nn.Conv2d(self.feat1_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.reduce2 = nn.Sequential(
            nn.Conv2d(self.feat2_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.reduce3 = nn.Sequential(
            nn.Conv2d(self.feat3_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Detection modules for different feature scales
        # Each module operates independently on its corresponding feature map
        # Input channels are 256 after reduction, output channels are 256
        self.detection_module1 = SSHDetectionModule(
            256, 256
        )  # M1: 256->256 channels for stride 4
        self.detection_module2 = SSHDetectionModule(
            256, 256
        )  # M2: 256->256 channels for stride 8
        self.detection_module3 = SSHDetectionModule(
            256, 256
        )  # M3: 256->256 channels for stride 16

        self.logger.debug("Model initialization complete")

    def forward(self, x):
        self.logger.debug("=" * 80)
        self.logger.debug(f"Input shape: {x.shape}")

        # Get backbone features (fused, c3, c4)
        self.logger.debug("Extracting backbone features...")
        with torch.no_grad():
            fused, c3, c4 = self.backbone(x)

        self.logger.debug("Backbone outputs:")
        self.logger.debug(
            f"  - Fused (stride 4): {fused.shape} (expected [B, 128, H/4, W/4])"
        )
        self.logger.debug(
            f"  - C3 (stride 8):    {c3.shape} (expected [B, 96, H/8, W/8])"
        )
        self.logger.debug(
            f"  - C4 (stride 16):   {c4.shape} (expected [B, 1280, H/16, W/16])"
        )

        # Apply dimension reduction to each feature map
        self.logger.debug("Applying dimension reduction...")
        f1 = self.reduce1(fused)  # stride 4, 128 -> 256 channels
        f2 = self.reduce2(c3)  # stride 8, 96 -> 256 channels
        f3 = self.reduce3(c4)  # stride 16, 1280 -> 256 channels

        self.logger.debug("After reduction:")
        self.logger.debug(
            f"  - f1 (stride 4):  {f1.shape} (from {fused.shape} -> [B, 256, H/4, W/4])"
        )
        self.logger.debug(
            f"  - f2 (stride 8):  {f2.shape} (from {c3.shape} -> [B, 256, H/8, W/8])"
        )
        self.logger.debug(
            f"  - f3 (stride 16): {f3.shape} (from {c4.shape} -> [B, 256, H/16, W/16]"
        )

        # Apply detection modules to each feature map independently
        self.logger.debug("Processing detection modules...")

        self.logger.debug("Detection Module 1 (stride 4):")
        cls1, reg1 = self.detection_module1(f1)

        self.logger.debug("Detection Module 2 (stride 8):")
        cls2, reg2 = self.detection_module2(f2)

        self.logger.debug("Detection Module 3 (stride 16):")
        cls3, reg3 = self.detection_module3(f3)

        self.logger.debug("Output shapes:")
        self.logger.debug(f"  - cls1: {cls1.shape}, reg1: {reg1.shape}")
        self.logger.debug(f"  - cls2: {cls2.shape}, reg2: {reg2.shape}")
        self.logger.debug(f"  - cls3: {cls3.shape}, reg3: {reg3.shape}")

        # Return in order of increasing stride (4, 8, 16)
        return [cls1, cls2, cls3], [reg1, reg2, reg3]


def train_epoch(
    model: SSH,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    anchors_list: List[Num[torch.Tensor, "1 N 4"]],
    epoch: int,
    max_grad_norm: float = 1.0,
):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    num_batches = 0

    level_weights = [1.0, 1.0, 1.0]

    logger.debug(f"Starting epoch {epoch}")
    logger.debug(f"Number of anchor sets: {len(anchors_list)}")
    for head_idx, anchors in enumerate(anchors_list):
        logger.debug(
            f"Anchor set {head_idx}: {anchors.shape} (device: {anchors.device})"
        )

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if not batch.get("bbox") or batch["image"].sum() == 0:
            print(f"Skipping corrupted batch {batch_idx}")
            continue

        imgs = batch["image"].to(device)

        gt_boxes = (
            torch.stack(batch["bbox"], dim=0).to(device)
            if batch["bbox"]
            else torch.zeros(0, 4, device=device)
        )

        logger.debug(f"\nProcessing batch {batch_idx}")
        logger.debug(f"Input images: {imgs.shape} (device: {imgs.device})")
        logger.debug(f"GT boxes: {gt_boxes.shape if gt_boxes is not None else 'None'}")

        cls_heads, reg_heads = model(imgs)

        logger.debug(f"Number of classification heads: {len(cls_heads)}")
        logger.debug(f"Number of regression heads: {len(reg_heads)}")

        batch_cls_loss = torch.tensor(0.0, device=device)
        batch_reg_loss = torch.tensor(0.0, device=device)

        for head_idx, (cls_pred, reg_pred) in enumerate(zip(cls_heads, reg_heads)):
            logger.debug(f"\nProcessing head {head_idx}")
            logger.debug(f"  cls_pred shape: {cls_pred.shape}")
            logger.debug(f"  reg_pred shape: {reg_pred.shape}")

            B, _, H, W = cls_pred.shape
            num_anchors = 2

            anchors = anchors_list[head_idx % len(anchors_list)].to(device)
            anchors = anchors.expand(B, -1, -1)
            logger.debug(f"  Anchors shape: {anchors.shape} (device: {anchors.device})")

            logger.debug("  Matching anchors to ground truth...")
            if gt_boxes.numel() == 0 or gt_boxes.shape[1] == 0:
                labels = torch.zeros(anchors.shape[:2], dtype=torch.long, device=device)
                matched_gt = torch.zeros_like(anchors)
            else:
                pos_thresh = 0.5 if head_idx < 2 else 0.4
                neg_thresh = 0.3
                labels, matched_gt = match_anchors_to_ground_truth(
                    anchors, gt_boxes, pos_iou=pos_thresh, neg_iou=neg_thresh
                )

            targets = encode_boxes(anchors, matched_gt)

            B, _, H, W = cls_pred.shape
            cls_pred = cls_pred.view(B, num_anchors, 2, H, W)
            cls_pred = cls_pred.permute(0, 3, 4, 1, 2).contiguous()
            cls_pred_flat = cls_pred.view(-1, 2)

            reg_pred = reg_pred.view(B, num_anchors, 4, H, W)
            reg_pred = reg_pred.permute(0, 3, 4, 1, 2).contiguous()
            reg_pred_flat = reg_pred.view(-1, 4)

            labels_flat = labels.view(-1)
            targets_flat = targets.view(-1, 4)

            logger.debug(f"Reshaped cls_pred: {cls_pred_flat.shape}")
            logger.debug(f"Reshaped reg_pred: {reg_pred_flat.shape}")
            logger.debug(f"Labels shape: {labels_flat.shape}")
            logger.debug(f"Targets shape: {targets_flat.shape}")

            # Focal loss parameters
            alpha = 0.25
            gamma = 2.0
            
            # Calculate cross entropy loss
            ce_loss = F.cross_entropy(
                cls_pred_flat, labels_flat, reduction='none', ignore_index=-1
            )
            
            # Calculate pt and focal weight
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** gamma
            
            # Apply class balancing
            alpha_factor = torch.ones_like(labels_flat) * alpha
            alpha_factor = torch.where(
                labels_flat == 1, alpha_factor, 1.0 - alpha_factor
            )
            
            # Combine focal loss components
            cls_loss = (alpha_factor * focal_weight * ce_loss).mean()

            pos_mask = labels_flat == 1
            reg_loss = torch.tensor(0.0, device=device)
            if pos_mask.sum() > 0:
                reg_loss = F.smooth_l1_loss(
                    reg_pred_flat[pos_mask], targets_flat[pos_mask], reduction="sum"
                ) / max(1, pos_mask.sum())

            weight = level_weights[head_idx] if head_idx < len(level_weights) else 1.0
            batch_cls_loss += cls_loss * weight
            batch_reg_loss += reg_loss * weight

        # Total loss with L2 regularization (handled by AdamW)
        loss = batch_cls_loss + batch_reg_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=1.0,
            norm_type=2
        )
        
        optimizer.step()
        
        # Log gradients for debugging
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Log metrics
        total_loss += loss.item()
        total_cls_loss += batch_cls_loss.item()
        total_reg_loss += batch_reg_loss.item()
        num_batches += 1

        if num_batches % 10 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / num_batches:.4f}",
                    "cls": f"{total_cls_loss / num_batches:.4f}",
                    "reg": f"{total_reg_loss / num_batches:.4f}",
                }
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0.0
    avg_reg_loss = total_reg_loss / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_cls_loss, avg_reg_loss


def validate(
    model: SSH,
    val_loader: DataLoader,
    anchors_list: List[Num[torch.Tensor, "1 N 4"]],
    epoch: int,
):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    # Loss weights for different feature levels (same as training)
    level_weights = [1.0, 1.0, 1.0]

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if not batch.get("bbox") or batch["image"].sum() == 0:
                print(f"Skipping corrupted batch {batch_idx}")
                continue

            imgs = batch["image"].to(device)

            gt_boxes = (
                torch.stack(batch["bbox"], dim=0).to(device)
                if batch["bbox"]
                else torch.zeros(0, 4, device=device)
            )

            # Forward pass
            cls_heads, reg_heads = model(imgs)

            batch_cls_loss = torch.tensor(0.0, device=device)
            batch_reg_loss = torch.tensor(0.0, device=device)

            # Process each detection head
            for head_idx, (cls_pred, reg_pred) in enumerate(zip(cls_heads, reg_heads)):
                B, _, H, W = cls_pred.shape
                num_anchors = 2

                # # Reshape predictions to match training
                # # cls_pred shape: [B, num_anchors*2, H, W]
                # # reg_pred shape: [B, num_anchors*4, H, W]

                # # Reshape cls_pred to [B, H, W, num_anchors, 2] -> [B, H*W*num_anchors, 2]
                # cls_pred = cls_pred.permute(
                #     0, 2, 3, 1
                # ).contiguous()  # [B, H, W, num_anchors*2]
                # cls_pred = cls_pred.view(
                #     B, -1, num_anchors, 2
                # )  # [B, H*W, num_anchors, 2]
                # cls_pred = cls_pred.reshape(B, -1, 2)  # [B, H*W*num_anchors, 2]

                # # Reshape reg_pred to [B, H, W, num_anchors, 4] -> [B, H*W*num_anchors, 4]
                # reg_pred = reg_pred.permute(
                #     0, 2, 3, 1
                # ).contiguous()  # [B, H, W, num_anchors*4]
                # reg_pred = reg_pred.view(
                #     B, -1, num_anchors, 4
                # )  # [B, H*W, num_anchors, 4]
                # reg_pred = reg_pred.reshape(B, -1, 4)  # [B, H*W*num_anchors, 4]

                # Get anchors for this level
                anchors = anchors_list[head_idx % len(anchors_list)]
                anchors = anchors.expand(B, -1, -1)

                # Match anchors to ground truth
                if gt_boxes.numel() == 0 or gt_boxes.shape[1] == 0:
                    labels = torch.zeros(
                        anchors.shape[:2], dtype=torch.long, device=device
                    )
                    matched_gt = torch.zeros_like(anchors)
                else:
                    # Use the same thresholds as in training
                    pos_thresh = 0.5 if head_idx < 2 else 0.4
                    neg_thresh = 0.3
                    labels, matched_gt = match_anchors_to_ground_truth(
                        anchors, gt_boxes, pos_iou=pos_thresh, neg_iou=neg_thresh
                    )

                # Encode targets
                targets = encode_boxes(anchors, matched_gt)

                # Flatten predictions and targets
                # cls_pred shape: [B, H*W*num_anchors, 2]
                # reg_pred shape: [B, H*W*num_anchors, 4]
                # labels shape: [B, H*W*num_anchors]
                # targets shape: [B, H*W*num_anchors, 4]
                cls_pred = cls_pred.view(B, num_anchors, 2, H, W)
                cls_pred = cls_pred.permute(0, 3, 4, 1, 2).contiguous()
                cls_pred_flat = cls_pred.view(-1, 2)

                reg_pred = reg_pred.view(B, num_anchors, 4, H, W)
                reg_pred = reg_pred.permute(0, 3, 4, 1, 2).contiguous()
                reg_pred_flat = reg_pred.view(-1, 4)

                labels_flat = labels.reshape(-1)
                targets_flat = targets.reshape(-1, 4)

                # Debug shapes
                if (
                    batch_idx == 0 and head_idx == 0
                ):  # Only log for first batch and head
                    logger.debug(
                        f"[Validation] Reshaped cls_pred: {cls_pred_flat.shape}"
                    )
                    logger.debug(
                        f"[Validation] Reshaped reg_pred: {reg_pred_flat.shape}"
                    )
                    logger.debug(f"[Validation] Labels shape: {labels_flat.shape}")
                    logger.debug(f"[Validation] Targets shape: {targets_flat.shape}")

                # Focal loss parameters
                alpha = 0.25
                gamma = 2.0
                
                # Calculate cross entropy loss
                ce_loss = F.cross_entropy(
                    cls_pred_flat, labels_flat, reduction='none', ignore_index=-1
                )
                
                # Calculate pt and focal weight
                pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** gamma
                
                # Apply class balancing
                alpha_factor = torch.ones_like(labels_flat) * alpha
                alpha_factor = torch.where(
                    labels_flat == 1, alpha_factor, 1.0 - alpha_factor
                )
                
                # Combine focal loss components
                cls_loss = (alpha_factor * focal_weight * ce_loss).mean()

                # Regression loss (only for positive samples)
                pos_mask = labels_flat == 1
                reg_loss = torch.tensor(0.0, device=device)
                if pos_mask.sum() > 0:
                    reg_loss = F.smooth_l1_loss(
                        reg_pred_flat[pos_mask], targets_flat[pos_mask], reduction="sum"
                    ) / max(1, pos_mask.sum())

                # Apply level weights
                weight = (
                    level_weights[head_idx] if head_idx < len(level_weights) else 1.0
                )
                batch_cls_loss += cls_loss * weight
                batch_reg_loss += reg_loss * weight

            # Total loss
            loss = batch_cls_loss + batch_reg_loss

            # Update metrics
            total_loss += loss.item()
            total_cls_loss += batch_cls_loss.item()
            total_reg_loss += batch_reg_loss.item()
            num_batches += 1

            # Debug shapes
            if batch_idx == 0:  # Only log for first batch
                logger.debug(f"cls_pred_flat shape: {cls_pred_flat.shape}")
                logger.debug(f"labels_flat shape: {labels_flat.shape}")
                logger.debug(f"reg_pred_flat shape: {reg_pred_flat.shape}")
                logger.debug(f"targets_flat shape: {targets_flat.shape}")
                logger.debug(f"pos_mask sum: {pos_mask.sum().item()}")
                logger.debug(f"cls_loss: {cls_loss.mean().item():.4f}")
                logger.debug(
                    f"reg_loss: {reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss:.4f}"
                )

            # Update progress bar
            if num_batches % 10 == 0:
                pbar.set_postfix(
                    {
                        "val_loss": f"{total_loss / num_batches:.4f}",
                        "val_cls": f"{total_cls_loss / num_batches:.4f}",
                        "val_reg": f"{total_reg_loss / num_batches:.4f}",
                    }
                )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0.0
    avg_reg_loss = total_reg_loss / num_batches if num_batches > 0 else 0.0

    logger.info(
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
    logger.info("Starting test_and_visualize")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of anchor sets: {len(anchors_list)}")
    for i, anchors in enumerate(anchors_list):
        logger.info(f"Anchor set {i} shape: {anchors.shape}")
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Color map for different feature levels (aligned with original SSH paper)
    colors = [
        (255, 255, 0),  # Green for stride 8 features (small faces)
        (0, 255, 255),  # Yellow for stride 16 features (medium faces)
        (255, 255, 0),  # Cyan for stride 32 features (large faces)
    ]

    # Score thresholds for each level (from paper: 0.5 for all levels)
    score_thresholds = [0.5, 0.5, 0.5]

    image_counter = 0  # Global counter for output filenames

    with torch.no_grad():
        logger.info("Starting inference loop")
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc="Testing", leave=False)
        ):
            logger.info(f"\n--- Processing batch {batch_idx} ---")
            if n_out is not None and image_counter >= n_out:
                logger.info(f"Reached maximum output limit of {n_out}, stopping")
                break

            logger.info(f"Batch keys: {list(batch.keys())}")
            if "image" not in batch:
                logger.warning(f"Skipping batch {batch_idx}: no 'image' key in batch")
                continue

            batch_size = batch["image"].shape[0]
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Input image shape: {batch['image'].shape}")

            # Forward pass
            img_tensor = batch["image"].to(device)
            logger.info(
                f"Input tensor shape: {img_tensor.shape}, device: {img_tensor.device}"
            )

            cls_heads, reg_heads = model(img_tensor)
            logger.info(f"Number of classification heads: {len(cls_heads)}")
            logger.info(f"Number of regression heads: {len(reg_heads)}")

            for i, (cls_head, reg_head) in enumerate(zip(cls_heads, reg_heads)):
                logger.info(
                    f"Head {i} - Cls shape: {cls_head.shape}, Reg shape: {reg_head.shape}"
                )

            # Process each image in the batch
            for b in range(batch_size):
                if n_out is not None and image_counter >= n_out:
                    break

                logger.info(f"\n=== Processing image {b} in batch {batch_idx} ===")

                # Prepare image for visualization
                img = batch["image"][b].cpu().numpy().transpose(1, 2, 0)
                logger.info(f"Transposed image shape: {img.shape}")
                img = (img * 255).astype(np.uint8)
                img = img.copy()
                logger.info(f"Final image shape: {img.shape}, dtype: {img.dtype}")

                # Draw ground truth boxes in red
                gt_count = 0
                if (
                    "bbox" in batch
                    and batch["bbox"] is not None
                    and len(batch["bbox"]) > b
                ):
                    gt_boxes = batch["bbox"][b]
                    if isinstance(gt_boxes, torch.Tensor):
                        gt_boxes = gt_boxes.cpu().numpy()
                    if len(gt_boxes) > 0:
                        logger.info(
                            f"Found {len(gt_boxes)} ground truth boxes for image {b}"
                        )
                        for box in gt_boxes:
                            x, y, w, h = box.astype(int)
                            pt1 = (int(x), int(y))
                            pt2 = (int(x + w), int(y + h))
                            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
                            gt_count += 1
                        logger.info(f"Drew {gt_count} ground truth boxes")

                all_boxes = []
                all_scores = []
                all_levels = []
                total_proposals = 0

                # Process each detection head (3 levels: stride 8, 16, 32)
                logger.info("Processing detection heads...")
                for head_idx, (cls_pred, reg_pred) in enumerate(
                    zip(cls_heads, reg_heads)
                ):
                    B, _, H, W = cls_pred.shape
                    num_anchors = 2
                    logger.info(f"\nProcessing head {head_idx}:")
                    logger.info(
                        f"  Input shapes - Cls: {cls_pred.shape}, Reg: {reg_pred.shape}"
                    )
                    logger.info(
                        f"  Spatial dims - H: {H}, W: {W}, Num anchors: {num_anchors}"
                    )

                    # Reshape predictions
                    cls_pred_reshaped = cls_pred.view(B, num_anchors, 2, H, W)
                    cls_pred_flat = cls_pred_reshaped.permute(0, 3, 4, 1, 2).reshape(
                        B, -1, 2
                    )

                    reg_pred_reshaped = reg_pred.view(B, num_anchors, 4, H, W)
                    reg_pred_flat = reg_pred_reshaped.permute(0, 3, 4, 1, 2).reshape(
                        B, -1, 4
                    )

                    logger.info(f"  After reshape:")
                    logger.info(
                        f"    Cls: {cls_pred_flat.shape} (from {cls_pred.shape})"
                    )
                    logger.info(
                        f"    Reg: {reg_pred_flat.shape} (from {reg_pred.shape})"
                    )

                    # Get anchors for this level
                    anchors = anchors_list[head_idx % len(anchors_list)]
                    anchors = anchors.expand(B, -1, -1)

                    # Decode boxes
                    logger.info("  Decoding boxes...")
                    boxes = decode_boxes(anchors, reg_pred_flat)
                    scores = F.softmax(cls_pred_flat, dim=2)[..., 1]
                    logger.info(
                        f"  Decoded boxes shape: {boxes.shape}, scores shape: {scores.shape}"
                    )
                    logger.info(
                        f"  Score range: {scores.min().item():.4f} to {scores.max().item():.4f}"
                    )

                    # Filter boxes for current image in batch
                    score_threshold = 0.1
                    mask = scores[b] > score_threshold  # shape [N]

                    filtered_boxes = boxes[b, mask]  # shape [num_kept, 4]
                    filtered_scores = scores[b, mask]  # shape [num_kept]

                    logger.info(
                        f"  After score threshold ({score_threshold}): {len(filtered_boxes)} boxes"
                    )

                    if len(filtered_boxes) > 0:
                        all_boxes.append(filtered_boxes.cpu().numpy())
                        all_scores.append(filtered_scores.cpu().numpy())
                        all_levels.extend([head_idx] * len(filtered_boxes))
                        total_proposals += len(filtered_boxes)
                        logger.info(
                            f"  Added {len(filtered_boxes)} proposals from head {head_idx}"
                        )
                    else:
                        logger.info(f"  No boxes passed threshold for head {head_idx}")

                logger.info(f"\nTotal proposals before NMS: {total_proposals}")

                # Process detections
                if all_boxes:
                    logger.info("Processing detections...")
                    try:
                        all_boxes = (
                            np.concatenate(all_boxes, axis=0)
                            if len(all_boxes) > 1
                            else all_boxes[0]
                        )
                        all_scores = (
                            np.concatenate(all_scores, axis=0)
                            if len(all_scores) > 1
                            else all_scores[0]
                        )
                        all_levels = np.array(all_levels)
                        logger.info(
                            f"After concatenation - Boxes: {all_boxes.shape}, Scores: {all_scores.shape}, Levels: {all_levels.shape}"
                        )
                    except Exception as e:
                        logger.error(f"Error concatenating detections: {e}")
                        logger.error(f"Box shapes: {[box.shape for box in all_boxes]}")
                        logger.error(
                            f"Score shapes: {[score.shape for score in all_scores]}"
                        )
                        continue

                    final_boxes = []
                    final_scores = []
                    final_levels = []

                    logger.info("Applying per-level NMS...")
                    for level in range(3):  # Only 3 levels now
                        level_mask = all_levels == level
                        if not np.any(level_mask):
                            logger.info(f"  Level {level}: No detections")
                            continue

                        level_boxes = all_boxes[level_mask]
                        level_scores = all_scores[level_mask]

                        logger.info(
                            f"  Level {level}: {len(level_boxes)} detections before NMS"
                        )

                        if len(level_boxes) == 0:
                            logger.info(f"  Level {level}: No boxes after filtering")
                            continue

                        # Apply NMS for this level
                        logger.info(
                            f"  Level {level}: Applying NMS to {len(level_boxes)} boxes"
                        )
                        try:
                            keep = nms(
                                torch.tensor(level_boxes, device=device),  # [N, 4]
                                torch.tensor(level_scores, device=device),  # [N]
                                0.3,  # NMS threshold from paper
                            )
                            logger.info(
                                f"  Level {level}: NMS kept {len(keep)}/{len(level_boxes)} boxes"
                            )

                            keep_indices = keep.cpu().numpy()
                            level_boxes = level_boxes[keep_indices]
                            level_scores = level_scores[keep_indices]
                        except Exception as e:
                            logger.error(f"Error in NMS for level {level}: {e}")
                            logger.error(
                                f"Boxes shape: {level_boxes.shape}, Scores shape: {level_scores.shape}"
                            )
                            continue

                        # Apply level-specific score threshold
                        threshold = score_thresholds[level]
                        keep = level_scores >= threshold
                        level_boxes = level_boxes[keep]
                        level_scores = level_scores[keep]
                        logger.info(
                            f"  Level {level}: After score threshold ({threshold}): {len(level_boxes)} boxes"
                        )

                        if len(level_boxes) > 0:
                            final_boxes.append(level_boxes)
                            final_scores.append(level_scores)
                            final_levels.extend([level] * len(level_boxes))
                            logger.info(
                                f"  Level {level}: Added {len(level_boxes)} final detections"
                            )
                        else:
                            logger.info(
                                f"  Level {level}: No detections after thresholding"
                            )

                    if final_boxes:
                        try:
                            final_boxes = (
                                np.concatenate(final_boxes, axis=0)
                                if len(final_boxes) > 1
                                else final_boxes[0]
                            )
                            final_scores = (
                                np.concatenate(final_scores, axis=0)
                                if len(final_scores) > 1
                                else final_scores[0]
                            )
                            final_levels = np.array(final_levels)
                            logger.info(f"Final detections: {len(final_boxes)} boxes")
                        except Exception as e:
                            logger.error(f"Error concatenating final detections: {e}")
                            continue

                        # Draw detections
                        logger.info(f"Drawing {len(final_boxes)} detections...")
                        drawn_boxes = 0
                        for box, score, level in zip(
                            final_boxes, final_scores, final_levels
                        ):
                            try:
                                x, y, w, h = box.astype(int)
                                x1, y1 = x, y
                                x2, y2 = x + w, y + h

                                color = colors[level % 3]  # Only 3 levels now

                                # Draw box
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                                # Draw level indicator
                                cv2.rectangle(
                                    img, (x1, y1 - 15), (x1 + 15, y1), color, -1
                                )

                                # Draw score
                                cv2.putText(
                                    img,
                                    f"{score:.2f}",
                                    (x1 + 20, y1 - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                    cv2.LINE_AA,
                                )
                                drawn_boxes += 1
                            except Exception as e:
                                logger.error(f"Error drawing box {drawn_boxes}: {e}")
                                logger.error(
                                    f"Box: {box}, Score: {score}, Level: {level}"
                                )
                        logger.info(f"Successfully drew {drawn_boxes} detections")

                # Save result
                output_path = os.path.join(
                    output_dir, f"test_result_{image_counter}.jpg"
                )
                try:
                    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    logger.info(f"Saved result to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving image {output_path}: {e}")

                image_counter += 1

    logger.info("Finished test_and_visualize")
