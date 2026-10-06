"""
extract_split_features_colab.py — Colab-specific version of
extract_split_features.py. Same hardcoded Drive layout as
build_gallery_colab.py (raw_images lives directly under MyDrive, not
nested inside the project folder).

Run from anywhere in Colab:
    python extract_split_features_colab.py --split val
    python extract_split_features_colab.py --split test
"""
import os
import sys
import time
import shutil
import argparse
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- This project's actual Drive layout (same as build_gallery_colab.py) ---
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/serbia_geolocalization"
DRIVE_RAW_IMAGES = "/content/drive/MyDrive/raw_images"
LOCAL_RAW_IMAGES = "/content/raw_images"

sys.path.append(DRIVE_PROJECT_ROOT)
from geotoken.encoder_pretraining.image_encoder import VisualEncoder


class SerbiaImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id"])
        candidates = [
            os.path.join(self.img_dir, f"{img_id}.jpg"),
            os.path.join(self.img_dir, f"{img_id}.jpeg"),
            os.path.join(self.img_dir, f"{img_id}.png"),
            os.path.join(self.img_dir, img_id),
        ]
        full_path = next((p for p in candidates if os.path.exists(p)), None)
        if full_path is not None:
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        return image, idx


def collate_pil_images(batch):
    images = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    return images, indices


def sync_images_to_local(src_dir: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[SYNC] Checking {src_dir} -> {dest_dir} ...")
    src_files = set(os.listdir(src_dir))
    existing = set(os.listdir(dest_dir))
    to_copy = src_files - existing
    if not to_copy:
        print(f"[SYNC] {dest_dir} already has all {len(src_files)} files. Skipping copy.")
        return
    print(f"[SYNC] Copying {len(to_copy)} new files...")
    for fname in tqdm(to_copy, desc="Syncing images"):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))
    print(f"[SYNC] Done. {len(os.listdir(dest_dir))} files now in {dest_dir}")


def sanity_check_images(df: pd.DataFrame, img_dir: str, n_check: int = 20):
    checked_ids = []
    found = 0
    for _, row in df.head(n_check).iterrows():
        img_id = str(row["id"])
        checked_ids.append(img_id)
        candidates = [
            os.path.join(img_dir, f"{img_id}.jpg"),
            os.path.join(img_dir, f"{img_id}.jpeg"),
            os.path.join(img_dir, f"{img_id}.png"),
            os.path.join(img_dir, img_id),
        ]
        if any(os.path.exists(p) for p in candidates):
            found += 1
    if found == 0:
        print(f"\n[FATAL] 0/{n_check} sample images resolved in: {img_dir}")
        print(f"  Sample ids from CSV: {checked_ids[:5]}")
        if os.path.isdir(img_dir):
            print(f"  Actual files found there (first 5): {os.listdir(img_dir)[:5]}")
        else:
            print(f"  That directory does not exist at all.")
        sys.exit(1)
    print(f"[OK] Pre-flight check: {found}/{n_check} sample images resolved in {img_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = parser.parse_args()

    print("=" * 65)
    print(f"[FEATURES] Extracting CLIP embeddings for split: {args.split}")
    print("=" * 65)

    processed_dir = os.path.join(DRIVE_PROJECT_ROOT, "data", "processed")
    csv_path = os.path.join(DRIVE_PROJECT_ROOT, "data", "osv5m_serbia_metadata.csv")
    out_dir = os.path.join(DRIVE_PROJECT_ROOT, "geotoken", "context_retrieval")

    sync_images_to_local(DRIVE_RAW_IMAGES, LOCAL_RAW_IMAGES)

    df_all = pd.read_csv(csv_path)
    all_s2_tokens = torch.load(os.path.join(processed_dir, "s2_tokens.pt"), weights_only=True)
    split_indices = torch.load(
        os.path.join(processed_dir, "splits", f"{args.split}_indices.pt"), weights_only=True
    )

    df_split = df_all.iloc[split_indices.tolist()].copy().reset_index(drop=True)
    split_s2_tokens = all_s2_tokens[split_indices]
    split_coords = torch.tensor(df_split[["latitude", "longitude"]].values, dtype=torch.float32)

    print(f"Split size: {len(df_split)}")

    sanity_check_images(df_split, LOCAL_RAW_IMAGES)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        print("\n[WARNING] No GPU detected. Running on CPU will be much slower.")

    print(f"Running inference on device: {device}")
    encoder = VisualEncoder(device=device)

    dataset = SerbiaImageDataset(df_split, LOCAL_RAW_IMAGES)
    batch_size = 64 if device != "cpu" else 32
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_pil_images,
    )

    all_embeddings = []
    n_missing = 0
    start_time = time.time()

    for i, (images, indices) in enumerate(tqdm(loader, desc=f"{args.split} embeddings")):
        for idx in indices:
            img_id = str(df_split.iloc[idx]["id"])
            candidates = [
                os.path.join(LOCAL_RAW_IMAGES, f"{img_id}.jpg"),
                os.path.join(LOCAL_RAW_IMAGES, f"{img_id}.jpeg"),
                os.path.join(LOCAL_RAW_IMAGES, f"{img_id}.png"),
                os.path.join(LOCAL_RAW_IMAGES, img_id),
            ]
            if not any(os.path.exists(p) for p in candidates):
                n_missing += 1

        with torch.no_grad():
            emb = encoder(images)
            all_embeddings.append(emb.cpu())

        if i == 2:
            elapsed = time.time() - start_time
            eta_min = (elapsed / 3) * len(loader) / 60.0
            print(f"\n[ESTIMATE] ~{elapsed/3:.2f}s/batch -> projected total: ~{eta_min:.1f} min")

    embeddings = torch.cat(all_embeddings, dim=0)

    if n_missing > 0:
        print(f"\n[WARNING] {n_missing}/{len(df_split)} images missing, encoded as blanks.")

    out_path = os.path.join(out_dir, f"{args.split}_features.pt")
    torch.save(
        {
            "embeddings": embeddings,
            "s2_tokens": split_s2_tokens,
            "coords": split_coords,
            "split_indices": split_indices,
            "n_missing_images": n_missing,
        },
        out_path,
    )

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"[DONE] Saved {args.split} features to: {out_path}")
    print(f"  - Embeddings: {embeddings.shape}")
    print(f"  - Total time: {total_time/60:.1f} minutes")
    print(f"  - Missing images: {n_missing}")
    print("=" * 65)


if __name__ == "__main__":
    main()
