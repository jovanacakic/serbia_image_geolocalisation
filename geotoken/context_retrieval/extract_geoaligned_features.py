"""
geotoken/context_retrieval/extract_geoaligned_features.py — replace GeoToken's
plain-CLIP image features with geo-aligned ones, closing the largest deviation
between our GeoToken implementation and the paper.

GeoToken's "Encoder Block" (their Fig. 2, right panel) is not a bare vision
encoder. It concatenates three views of the image:

    e^image = concat(e^img_raw, e^image_text, e^image_gps)

where the latter two come from the geo-alignment pretraining of their Section
III-B, which they take from G3. Our implementation used only e^img_raw, so
GeoToken was the one model in this project whose features carried no
location-aware training --- while both comparison models used geo-aligned
features. That asymmetry understated GeoToken.

We already have such an encoder: the image tower of our trained G3
geo-alignment model. This script re-extracts every split's embeddings as

    concat( raw CLIP (768-d),  G3 geo-aligned (512-d) )  ->  1280-d

Each half is L2-normalised before concatenation so neither dominates; the
retrieval searcher renormalises the whole vector anyway. This is a two-view
approximation of the paper's three-view concatenation, since our G3 uses one
shared image projection rather than separate text- and GPS-aligned ones.

Only the `embeddings` field changes. S2 tokens, coordinates and split indices
are copied verbatim from the existing feature files, so row order is preserved
exactly and nothing else in the pipeline shifts. Outputs are written to new
`geoaligned_*` files, leaving the originals untouched for comparison.

Usage (Colab, after %cd into the project root):
    !python geotoken/context_retrieval/extract_geoaligned_features.py \
        --g3_checkpoint g3_adaptation/checkpoints/g3_text_v2/best_model.pt
"""
import os
import sys
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from g3_adaptation.model import G3GeoAlignment
from geoclip_adaptation.dataset import EVAL_TRANSFORM
from geoclip_adaptation.train_lora import (
    sync_images_to_local, DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
from utils.checkpoint_utils import load_any_state_dict

CSV_PATH = os.path.join(BASE_DIR, "data", "osv5m_serbia_metadata.csv")
CTX_DIR = os.path.join(BASE_DIR, "geotoken", "context_retrieval")

# (source file, key holding that file's row indices into the metadata CSV)
TARGETS = [
    ("gallery_database.pt", "train_indices"),
    ("val_features.pt", "split_indices"),
    ("test_features.pt", "split_indices"),
]


class IndexedImageDataset(Dataset):
    """Loads images in the exact row order of the feature file being rebuilt,
    so the new embeddings line up with the existing tokens and coordinates.

    Returns [V, 3, 224, 224]: one tensor per test-time-augmentation view, or a
    single view when TTA is off. Each image is read from disk once regardless
    of V, so TTA costs GPU time, not extra I/O.
    """

    def __init__(self, df, img_dir, views=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.views = views or [("eval", EVAL_TRANSFORM)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img_id = str(self.df.iloc[i]["id"])
        path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = (Image.open(path).convert("RGB") if os.path.exists(path)
               else Image.new("RGB", (224, 224), (0, 0, 0)))
        return torch.stack([t(img) for _, t in self.views])


@torch.no_grad()
def encode_split(model, df, img_dir, device, batch_size, num_workers, use_amp,
                 views=None):
    loader = DataLoader(IndexedImageDataset(df, img_dir, views),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for stack in tqdm(loader, desc="    encoding"):
        b, v = stack.shape[:2]
        px = stack.view(b * v, *stack.shape[2:]).to(device)
        with autocast("cuda", enabled=use_amp):
            # Raw CLIP view: the same 768-d features GeoToken used before.
            raw = model.image_encoder.CLIP.get_image_features(pixel_values=px)
            # Geo-aligned view: 768 -> MLP -> 512, trained contrastively
            # against GPS and text.
            aligned = model.encode_image(px)
        # Normalise per view first, so no single view dominates the average
        # through sheer magnitude, then average, then renormalise: averaging
        # unit vectors denormalises them, and the retrieval searcher expects
        # each half to be a direction rather than a direction-and-length.
        raw = F.normalize(raw.float(), dim=-1).view(b, v, -1).mean(dim=1)
        aligned = F.normalize(aligned.float(), dim=-1).view(b, v, -1).mean(dim=1)
        raw = F.normalize(raw, dim=-1)
        aligned = F.normalize(aligned, dim=-1)
        out.append(torch.cat([raw, aligned], dim=-1).cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g3_checkpoint", type=str,
                    default="g3_adaptation/checkpoints/g3_text_v2/best_model.pt")
    ap.add_argument("--image_adaptation", choices=["frozen_mlp", "lora"], default="frozen_mlp")
    ap.add_argument("--no_text", action="store_true",
                    help="Set if the G3 checkpoint was trained with --no_text.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefix", type=str, default=None,
                    help="Output filename prefix. Defaults to 'geoaligned_', "
                         "or 'geoaligned_tta_' when --tta is set.")
    ap.add_argument("--tta", action="store_true",
                    help="Average the 6 fixed test-time-augmentation views used "
                         "by evaluate_test.py. Query splits only.")
    ap.add_argument("--splits", type=str, default=None,
                    help="Comma-separated subset of {gallery,val,test}. Defaults "
                         "to all three, or to 'val,test' when --tta is set.")
    args = ap.parse_args()

    views = None
    if args.tta:
        from evaluate_test import tta_views
        views = tta_views()

    if args.prefix is None:
        args.prefix = "geoaligned_tta_" if args.tta else "geoaligned_"

    alias = {"gallery": "gallery_database.pt", "val": "val_features.pt",
             "test": "test_features.pt"}
    if args.splits:
        wanted = {alias[s.strip()] for s in args.splits.split(",")}
    else:
        # The gallery is the retrieval database GeoToken was trained against.
        # Augmenting it would change the database out from under the trained
        # decoder, so TTA applies to queries only.
        wanted = {"val_features.pt", "test_features.pt"} if args.tta else set(alias.values())
    if args.tta and "gallery_database.pt" in wanted:
        ap.error("--tta must not be applied to the gallery: it is the retrieval "
                 "database the decoder was trained against, not a query set.")
    targets = [(f, k) for f, k in TARGETS if f in wanted]

    print("=" * 68)
    print("GEO-ALIGNED FEATURE EXTRACTION for GeoToken"
          + (f"  [TTA: {len(views)} views]" if args.tta else ""))
    print("=" * 68)
    print("Rebuilding: " + ", ".join(f for f, _ in targets))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"Device: {device}")

    ckpt = args.g3_checkpoint if os.path.isabs(args.g3_checkpoint) \
        else os.path.join(BASE_DIR, args.g3_checkpoint)
    assert os.path.exists(ckpt), f"G3 checkpoint not found: {ckpt}"

    print(f"Loading G3 encoder from {ckpt}")
    model = G3GeoAlignment(image_adaptation=args.image_adaptation,
                           use_text=not args.no_text).to(device)
    load_any_state_dict(model, torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    img_dir = sync_images_to_local(DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    df_all = pd.read_csv(CSV_PATH)

    for fname, index_key in targets:
        src = os.path.join(CTX_DIR, fname)
        if not os.path.exists(src):
            print(f"[skip] {fname} not found")
            continue

        data = torch.load(src, weights_only=True)
        if index_key not in data:
            print(f"[skip] {fname}: no '{index_key}' field to order rows by")
            continue

        idx = data[index_key]
        df = df_all.iloc[idx.tolist()]
        print(f"\n{fname}: {len(df)} images "
              f"(existing embeddings {tuple(data['embeddings'].shape)})")

        emb = encode_split(model, df, img_dir, device,
                           args.batch_size, args.num_workers, use_amp, views)
        assert emb.shape[0] == data["embeddings"].shape[0], (
            f"row count changed: {emb.shape[0]} vs {data['embeddings'].shape[0]}")

        new = dict(data)
        new["embeddings"] = emb
        out = os.path.join(CTX_DIR, args.prefix + fname)
        torch.save(new, out)
        print(f"  -> {os.path.basename(out)}  embeddings {tuple(emb.shape)}  "
              f"(tokens/coords copied unchanged)")

    print("\n" + "=" * 68)
    if args.tta:
        # No retraining: TTA changes only the query embeddings, and the trained
        # decoder and gallery are reused as they are.
        print("Done. Evaluate the existing checkpoint against these features:")
        print()
        print("  !python evaluate_test.py --model geotoken --split test \\")
        print("      --clip_dim 1280 --label geotoken_test_tta \\")
        print("      --checkpoint geotoken/transformer_model/checkpoints/"
              "loss_paper_plus_reg_geoalign/best_geotoken_model.pt \\")
        print("      --gallery_path geotoken/context_retrieval/"
              "geoaligned_gallery_database.pt \\")
        print(f"      --features_path geotoken/context_retrieval/"
              f"{args.prefix}test_features.pt")
        print()
        print("The gallery stays un-augmented on purpose: it is the retrieval")
        print("database the decoder was trained against.")
        print("=" * 68)
        return
    print("Done. Retrain GeoToken against these files, e.g.:")
    print()
    print("  !python geotoken/transformer_model/train.py \\")
    print("      --run_name loss_paper_plus_reg_geoalign --clip_dim 1280 \\")
    print(f"      --gallery_path geotoken/context_retrieval/{args.prefix}gallery_database.pt \\")
    print("      --loss_weighting paper --dropout 0.2 --weight_decay 2e-2 \\")
    print("      --lr 1e-4 --scheduler cosine --batch_size 64 --epochs 40")
    print()
    print("NOTE: train.py reads val features from val_features.pt by path, so")
    print(f"swap in {args.prefix}val_features.pt before training (see --help).")
    print("=" * 68)


if __name__ == "__main__":
    main()
