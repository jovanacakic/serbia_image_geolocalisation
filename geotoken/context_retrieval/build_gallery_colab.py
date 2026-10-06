"""
build_gallery_colab.py — Colab-specific version of build_gallery.py.

Hardcodes this project's actual Google Drive layout:
  - Project files (CSV, code, splits) live under:
        /content/drive/MyDrive/geotoken_serbia
  - Raw images live in a SEPARATE folder, directly under MyDrive (NOT
    nested under geotoken_serbia/data/raw_images):
        /content/drive/MyDrive/raw_images

What this script does, in order:
  1. Syncs raw_images from Drive to local Colab disk (/content/raw_images).
     Reading thousands of small files straight off a Drive mount is what
     made earlier runs take hours; local disk is close to instant. This
     step is resumable — re-running skips files already copied.
  2. Runs a pre-flight check on a small sample before committing GPU time.
  3. Builds gallery_database.pt from the current TRAIN split, exactly like
     build_gallery.py did.

Run from anywhere in Colab (adjust the two DRIVE_* paths below if your
folder names ever change):
    python build_gallery_colab.py
"""
import os
import sys
import shutil
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- This project's actual Drive layout ---
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/serbia_geolocalization"
DRIVE_RAW_IMAGES = "/content/drive/MyDrive/raw_images"
LOCAL_RAW_IMAGES = "/content/raw_images"

sys.path.append(DRIVE_PROJECT_ROOT)
from geotoken.encoder_pretraining.image_encoder import VisualEncoder


class SerbiaImageDataset(Dataset):
    """Loads images from disk matching the `id` column in metadata."""

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
    """Copy any files from src_dir not already present in dest_dir.
    Resumable: safe to re-run, only copies what's missing."""
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[SYNC] Checking {src_dir} -> {dest_dir} ...")

    src_files = set(os.listdir(src_dir))
    existing = set(os.listdir(dest_dir))
    to_copy = src_files - existing

    if not to_copy:
        print(f"[SYNC] {dest_dir} already has all {len(src_files)} files. Skipping copy.")
        return

    print(f"[SYNC] Copying {len(to_copy)} new files (this reads from Drive, "
          f"so it's the slow part — one-time cost, later runs will skip it).")
    for fname in tqdm(to_copy, desc="Syncing images"):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))
    print(f"[SYNC] Done. {len(os.listdir(dest_dir))} files now in {dest_dir}")


def sanity_check_images(df: pd.DataFrame, img_dir: str, n_check: int = 20):
    """Fail fast instead of silently encoding blanks for the whole dataset."""
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
        print("  Aborting before wasting GPU time on 100% blank images.")
        sys.exit(1)

    print(f"[OK] Pre-flight check: {found}/{n_check} sample images resolved in {img_dir}")


def main():
    print("=" * 65)
    print("[STEP 2] Building RAG Context Retrieval Gallery (Colab)")
    print("=" * 65)

    CSV_PATH = os.path.join(DRIVE_PROJECT_ROOT, "data", "osv5m_serbia_metadata.csv")
    PROCESSED_DIR = os.path.join(DRIVE_PROJECT_ROOT, "data", "processed")
    OUTPUT_GALLERY_PATH = os.path.join(
        DRIVE_PROJECT_ROOT, "geotoken", "context_retrieval", "gallery_database.pt"
    )

    print(f"CSV path: {CSV_PATH}")
    assert os.path.exists(CSV_PATH), f"CSV not found at {CSV_PATH} — check DRIVE_PROJECT_ROOT."

    # 0. Sync images to local disk first
    sync_images_to_local(DRIVE_RAW_IMAGES, LOCAL_RAW_IMAGES)

    # 1. Load metadata, tokens, and train split indices
    df_all = pd.read_csv(CSV_PATH)
    all_s2_tokens = torch.load(
        os.path.join(PROCESSED_DIR, "s2_tokens.pt"), weights_only=True
    )
    train_indices = torch.load(
        os.path.join(PROCESSED_DIR, "splits", "train_indices.pt"), weights_only=True
    )

    print(f"Total dataset entries : {len(df_all)}")
    print(f"Gallery target size   : {len(train_indices)} (Train split exclusively)")

    df_train = df_all.iloc[train_indices.tolist()].copy().reset_index(drop=True)
    gallery_s2_tokens = all_s2_tokens[train_indices]
    gallery_coords = torch.tensor(
        df_train[["latitude", "longitude"]].values, dtype=torch.float32
    )

    # 2. Pre-flight check
    sanity_check_images(df_train, LOCAL_RAW_IMAGES)

    # 3. Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Running inference on device: {device}")
    encoder = VisualEncoder(device=device)

    # 4. DataLoader
    dataset = SerbiaImageDataset(df_train, LOCAL_RAW_IMAGES)
    batch_size = 64 if device != "cpu" else 32
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_pil_images,
    )

    # 5. Extract embeddings
    all_embeddings = []
    n_missing = 0
    print(f"\n[EXTRACTING] Processing {len(dataset)} images in batches...")

    for images, indices in tqdm(loader, desc="Gallery Embeddings"):
        for idx in indices:
            img_id = str(df_train.iloc[idx]["id"])
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

    gallery_embeddings = torch.cat(all_embeddings, dim=0)

    if n_missing > 0:
        print(f"\n[WARNING] {n_missing}/{len(dataset)} images were missing "
              f"and got encoded as blank black images.")

    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_GALLERY_PATH), exist_ok=True)
    gallery_data = {
        "embeddings": gallery_embeddings,
        "s2_tokens": gallery_s2_tokens,
        "coords": gallery_coords,
        "train_indices": train_indices,
        "n_missing_images": n_missing,
    }
    torch.save(gallery_data, OUTPUT_GALLERY_PATH)

    print("\n" + "=" * 65)
    print(f"[GALLERY READY] Saved to: {OUTPUT_GALLERY_PATH}")
    print(f"  - Embeddings Matrix : {gallery_embeddings.shape}")
    print(f"  - S2 Tokens Matrix  : {gallery_s2_tokens.shape}")
    print(f"  - Coordinates Matrix: {gallery_coords.shape}")
    print(f"  - Missing images    : {n_missing}")
    print("=" * 65)


if __name__ == "__main__":
    main()
