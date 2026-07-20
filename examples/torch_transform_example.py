#!/usr/bin/env python3
# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from stainx import StainNormalizerTransform


class BatchToImage(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.to_image = v2.ToImage()
        self.device = device

    def forward(self, imgs):
        if isinstance(imgs, list):
            batch_tensor = torch.stack([self.to_image(img) for img in imgs])
            return batch_tensor.to(self.device) if self.device else batch_tensor
        return self.to_image(imgs).to(self.device) if self.device else self.to_image(imgs)


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return Image.open(self.image_paths[idx]).convert("RGB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    reference_img = Image.open(os.path.join(data_dir, "target.png")).convert("RGB")
    reference_tensor = v2.ToImage()(reference_img).to(device)

    stain_transform = StainNormalizerTransform(method="macenko", mode="reference", reference=reference_tensor, device=device)

    transforms = v2.Compose([BatchToImage(device=device), v2.ToDtype(torch.float32, scale=True), v2.RandomResizedCrop(size=(224, 224), antialias=True), stain_transform, v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_image_paths = [os.path.join(data_dir, f"test_{i}.png") for i in range(1, 6) if os.path.exists(os.path.join(data_dir, f"test_{i}.png"))]
    dataset = ImageDataset(test_image_paths)

    def collate_fn(batch):
        return transforms(list(batch))

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    for batch_idx, batch in enumerate(dataloader):
        print(f"Device: {device}, Batch {batch_idx + 1} shape: {batch.shape}, dtype: {batch.dtype}")


if __name__ == "__main__":
    main()
