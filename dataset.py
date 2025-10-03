from utils import (
    pad_to,
    scale_bbox,
)

from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import torch
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def collate_fn(batch):
    images = []
    bbox = []

    for item in batch:
        orig_size = item["image"].size
        padded_img = pad_to(item["image"])
        images.append(transform(padded_img))

        scaled_bbox = scale_bbox(item["faces"]["bbox"], orig_size)
        bbox.append(torch.tensor(scaled_bbox, dtype=torch.float32))

    images = torch.stack(images)
    return {
        "image": images,
        "bbox": bbox,
    }


def collate_fn_test(batch):
    images = []

    for item in batch:
        padded_img = pad_to(
            item["image"],
        )
        images.append(transform(padded_img))

    images = torch.stack(images)

    return {
        "image": images,
    }


ds_train = load_dataset("CUHK-CSE/wider_face", split="train", trust_remote_code=True)
ds_val = load_dataset("CUHK-CSE/wider_face", split="validation", trust_remote_code=True)
ds_test = load_dataset("CUHK-CSE/wider_face", split="test", trust_remote_code=True)
train_loader = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ds_val, batch_size=1, collate_fn=collate_fn)
test_loader = DataLoader(ds_test, batch_size=1, collate_fn=collate_fn_test)
