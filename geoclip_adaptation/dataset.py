"""
geoclip_adaptation/dataset.py — loads Serbia images + GPS targets for the
LoRA/regression fine-tuning of GeoCLIP, applying SimCLR-style augmentation
for training and a plain resize+normalize for val/test.

Uses the same sequence-grouped split index files GeoToken uses
(data/processed/splits/{split}_indices.pt) so results are comparable across
models — not a fresh re-split.
"""
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# OpenAI CLIP normalization constants.
CLIP_MEAN = [0.481, 0.458, 0.408]
CLIP_STD = [0.269, 0.261, 0.276]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


class SerbiaGeoCLIPDataset(Dataset):
    def __init__(self, csv_path, raw_images_dir, split_indices_path, train):
        df_all = pd.read_csv(csv_path)
        indices = torch.load(split_indices_path, weights_only=True)
        self.df = df_all.iloc[indices.tolist()].reset_index(drop=True)
        self.img_dir = raw_images_dir
        self.transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id"])
        path = os.path.join(self.img_dir, f"{img_id}.jpg")
        if os.path.exists(path):
            image = Image.open(path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        pixel_values = self.transform(image)
        coords = torch.tensor([row["latitude"], row["longitude"]], dtype=torch.float32)
        return pixel_values, coords
