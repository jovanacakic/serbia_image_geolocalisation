import os
import sys
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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

        # Check if file has an extension or needs .jpg appended
        candidates = [
            os.path.join(self.img_dir, f"{img_id}.jpg"),
            os.path.join(self.img_dir, f"{img_id}.jpeg"),
            os.path.join(self.img_dir, f"{img_id}.png"),
            os.path.join(self.img_dir, img_id),
        ]

        full_path = None
        for path in candidates:
            if os.path.exists(path):
                full_path = path
                break

        if full_path is not None:
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            # Fallback blank image if file is missing
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        return image, idx


def collate_pil_images(batch):
    images = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    return images, indices


def main():
    print("=" * 65)
    print("[STEP 2] Building RAG Context Retrieval Gallery...")
    print("=" * 65)

    DATA_DIR = "data"
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    CSV_PATH = os.path.join(DATA_DIR, "osv5m_serbia_metadata.csv")
    RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
    OUTPUT_GALLERY_PATH = "geotoken/context_retrieval/gallery_database.pt"

    # 1. Load metadata, tokens, and train split indices
    df_all = pd.read_csv(CSV_PATH)
    all_s2_tokens = torch.load(
        os.path.join(PROCESSED_DIR, "s2_tokens.pt"), weights_only=True
    )
    train_indices = torch.load(
        os.path.join(PROCESSED_DIR, "splits", "train_indices.pt"),
        weights_only=True,
    )

    print(f"Total dataset entries : {len(df_all)}")
    print(
        f"Gallery target size   : {len(train_indices)} (Train split exclusively)"
    )

    # 2. Filter exclusively for training set
    df_train = df_all.iloc[train_indices.tolist()].copy().reset_index(drop=True)
    gallery_s2_tokens = all_s2_tokens[train_indices]
    gallery_coords = torch.tensor(
        df_train[["latitude", "longitude"]].values, dtype=torch.float32
    )

    # 3. Setup hardware device (MPS for Apple Silicon Mac, CUDA for Nvidia, CPU fallback)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Running inference on device: {device}")
    encoder = VisualEncoder(device=device)

    # 4. DataLoader
    dataset = SerbiaImageDataset(df_train, RAW_IMAGES_DIR)
    batch_size = 64 if device != "cpu" else 32
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pil_images,
    )

    # 5. Extract Embeddings
    all_embeddings = []
    print(f"\n[EXTRACTING] Processing {len(dataset)} images in batches...")

    for images, _ in tqdm(loader, desc="Gallery Embeddings"):
        with torch.no_grad():
            emb = encoder(images)  # Shape: [B, 768]
            all_embeddings.append(emb.cpu())

    gallery_embeddings = torch.cat(all_embeddings, dim=0)

    # 6. Save Gallery File
    os.makedirs(os.path.dirname(OUTPUT_GALLERY_PATH), exist_ok=True)
    gallery_data = {
        "embeddings": gallery_embeddings,  # [16292, 768]
        "s2_tokens": gallery_s2_tokens,  # [16292, 21]
        "coords": gallery_coords,  # [16292, 2]
        "train_indices": train_indices,
    }

    torch.save(gallery_data, OUTPUT_GALLERY_PATH)

    print("\n" + "=" * 65)
    print(f"[GALLERY READY] Saved to: {OUTPUT_GALLERY_PATH}")
    print(f"  - Embeddings Matrix : {gallery_embeddings.shape}")
    print(f"  - S2 Tokens Matrix  : {gallery_s2_tokens.shape}")
    print(f"  - Coordinates Matrix: {gallery_coords.shape}")
    print("=" * 65)


if __name__ == "__main__":
    main()