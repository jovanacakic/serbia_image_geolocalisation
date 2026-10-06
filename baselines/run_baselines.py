"""
baselines/run_baselines.py — reference points that every adapted model
(GeoToken, GeoCLIP-regression, G3) is measured against, all scored through
the same metrics used everywhere else in this project (utils/eval_utils).

Four baselines, cheapest first:

  centroid        Always predict the training set's centroid, ignoring the
                  image entirely. The floor: any real model must beat this,
                  or it has learned nothing image-dependent.

  oracle_train    For each query, the closest point in the TRAIN split to the
                  TRUE location. The ceiling for retrieval over a dense
                  in-country gallery, which is what G3 uses. Contrast with
                  oracle_gallery below: the gap between the two is the cost
                  of GeoCLIP's sparse worldwide gallery, not of its encoder.

  oracle_gallery  For each query, the closest point in GeoCLIP's shipped
                  100K-point GPS gallery to the TRUE location. Not a
                  predictor — an upper bound on what ANY retrieval-based
                  method could achieve with this gallery, however perfect
                  its image encoder. Same idea as the "Ideal Selector"
                  diagnostic used for GeoToken's candidate pools.

  geoclip_global  Pretrained GeoCLIP, unmodified, retrieving over its full
                  worldwide gallery.

  geoclip_serbia  Same model, gallery restricted to points inside Serbia's
                  border polygon. The fair retrieval baseline: it gets the
                  same "the answer is in Serbia" prior our regression head
                  has by construction.

Only 70 of GeoCLIP's 100,000 gallery points fall inside Serbia's polygon, so
the retrieval ceiling (oracle_gallery) is severely limited by gallery density
rather than by encoder quality.

Usage (Colab, after %cd into the project root):
  !python baselines/run_baselines.py                      # all, on val
  !python baselines/run_baselines.py --baselines centroid,oracle_gallery
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path as MplPath
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from geoclip_adaptation.dataset import SerbiaGeoCLIPDataset
from geoclip_adaptation.train_lora import (
    sync_images_to_local, DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR,
)
from utils.eval_utils import haversine_distance, format_metrics

CSV_PATH = os.path.join(BASE_DIR, "data", "osv5m_serbia_metadata.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "processed", "splits")

# Serbia's border, as supplied for this project (lon, lat) pairs.
SERBIA_POLYGON = [
    [18.7927397, 45.9406893], [19.1637458, 44.9357462], [19.2155141, 43.5200325],
    [20.3371605, 42.8721743], [20.785819, 43.2566879], [21.7866727, 42.6694981],
    [21.5623434, 42.306839], [22.4769166, 42.3387353], [22.9773435, 43.1875276],
    [22.4510325, 43.6949599], [22.6753617, 44.2290549], [22.7357581, 44.5926913],
    [22.3561239, 44.7460926], [22.1663069, 44.4819911], [21.4415507, 45.209939],
    [20.2340787, 46.1284229], [19.5690201, 46.1831578], [18.7927397, 45.9406893],
]


def compute_metrics(pred_coords, true_coords):
    distances = haversine_distance(pred_coords, true_coords)
    metrics = {}
    for th in (1, 25, 200, 750, 2500):
        metrics[f"acc@{th}km"] = (distances <= th).float().mean().item() * 100
    metrics["median_km"] = distances.median().item()
    metrics["mean_km"] = distances.mean().item()
    return metrics


def load_split_coords(split):
    df_all = pd.read_csv(CSV_PATH)
    indices = torch.load(os.path.join(SPLITS_DIR, f"{split}_indices.pt"), weights_only=True)
    df = df_all.iloc[indices.tolist()].reset_index(drop=True)
    return torch.tensor(df[["latitude", "longitude"]].values, dtype=torch.float32)


def load_geoclip_gallery():
    import geoclip
    gallery_csv = os.path.join(
        os.path.dirname(geoclip.__file__), "model", "gps_gallery", "coordinates_100K.csv"
    )
    gal = pd.read_csv(gallery_csv)
    return torch.tensor(gal[["LAT", "LON"]].values, dtype=torch.float32)


def restrict_to_serbia(gallery):
    path = MplPath(SERBIA_POLYGON)
    pts = np.column_stack([gallery[:, 1].numpy(), gallery[:, 0].numpy()])  # (lon, lat)
    inside = path.contains_points(pts)
    return gallery[torch.from_numpy(inside)]


def nearest_gallery_point(true_coords, gallery, chunk=256):
    """Oracle: closest gallery entry to each ground-truth location. Chunked
    over queries so the 100K-point global gallery doesn't blow up memory."""
    out = []
    for start in range(0, true_coords.size(0), chunk):
        batch = true_coords[start:start + chunk]                     # [b, 2]
        d = haversine_distance(batch.unsqueeze(1), gallery.unsqueeze(0))  # [b, G]
        out.append(gallery[d.argmin(dim=1)])
    return torch.cat(out, dim=0)


@torch.no_grad()
def geoclip_retrieval(gallery, split, device, batch_size=32, num_workers=2):
    """Pretrained GeoCLIP, unmodified, retrieving over `gallery`."""
    from geoclip import GeoCLIP

    model = GeoCLIP(from_pretrained=True).to(device)
    model.eval()

    raw_images_dir = sync_images_to_local(DRIVE_RAW_IMAGES_DIR, LOCAL_RAW_IMAGES_DIR)
    ds = SerbiaGeoCLIPDataset(
        CSV_PATH, raw_images_dir, os.path.join(SPLITS_DIR, f"{split}_indices.pt"), train=False
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    gallery_dev = gallery.to(device)
    loc_features = model.location_encoder(gallery_dev)
    loc_features = torch.nn.functional.normalize(loc_features, dim=1)

    preds = []
    for pixel_values, _ in tqdm(loader, desc="  GeoCLIP retrieval"):
        img_features = model.image_encoder(pixel_values.to(device))
        img_features = torch.nn.functional.normalize(img_features, dim=1)
        logits = img_features @ loc_features.t()
        preds.append(gallery_dev[logits.argmax(dim=1)].cpu())
    return torch.cat(preds, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--baselines", type=str,
        default="centroid,oracle_train,oracle_gallery,geoclip_global,geoclip_serbia",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output_csv", type=str, default="baselines/baseline_results.csv",
        help="Appended to, so results accumulate into one comparable table.",
    )
    args = parser.parse_args()

    requested = [b.strip() for b in args.baselines.split(",")]

    print("=" * 65)
    print(f"BASELINES — split: {args.split}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    true_coords = load_split_coords(args.split)
    train_coords = load_split_coords("train")
    print(f"{args.split}: {true_coords.size(0)} images | train: {train_coords.size(0)} images")
    print(f"Device: {device}")

    results = {}

    if "centroid" in requested:
        print("\n[centroid] Predicting the training centroid for every image")
        centroid = train_coords.mean(dim=0)
        print(f"  Train centroid: lat={centroid[0]:.4f}, lon={centroid[1]:.4f}")
        preds = centroid.unsqueeze(0).expand(true_coords.size(0), 2)
        results["Centroid (train mean)"] = compute_metrics(preds, true_coords)

    if "oracle_train" in requested:
        print("\n[oracle_train] Closest TRAIN coordinate to ground truth "
              "(ceiling for retrieval over a dense in-country gallery)")
        results["Oracle: train gallery"] = compute_metrics(
            nearest_gallery_point(true_coords, train_coords), true_coords
        )

    gallery = None
    if any(b in requested for b in ("oracle_gallery", "geoclip_global", "geoclip_serbia")):
        gallery = load_geoclip_gallery()
        serbia_gallery = restrict_to_serbia(gallery)
        print(f"\nGeoCLIP gallery: {gallery.size(0)} points worldwide, "
              f"{serbia_gallery.size(0)} inside Serbia's polygon")

    if "oracle_gallery" in requested:
        print("\n[oracle_gallery] Closest gallery point to ground truth "
              "(retrieval ceiling, not a predictor)")
        results["Oracle: global gallery"] = compute_metrics(
            nearest_gallery_point(true_coords, gallery), true_coords
        )
        results["Oracle: Serbia gallery"] = compute_metrics(
            nearest_gallery_point(true_coords, serbia_gallery), true_coords
        )

    if "geoclip_global" in requested:
        print("\n[geoclip_global] Pretrained GeoCLIP, full worldwide gallery")
        preds = geoclip_retrieval(gallery, args.split, device, args.batch_size, args.num_workers)
        results["GeoCLIP zero-shot (global gallery)"] = compute_metrics(preds, true_coords)

    if "geoclip_serbia" in requested:
        print("\n[geoclip_serbia] Pretrained GeoCLIP, Serbia-restricted gallery")
        preds = geoclip_retrieval(serbia_gallery, args.split, device, args.batch_size, args.num_workers)
        results["GeoCLIP zero-shot (Serbia gallery)"] = compute_metrics(preds, true_coords)

    print("\n" + "=" * 65)
    print(f"RESULTS ({args.split} set, {true_coords.size(0)} images)")
    print("=" * 65)
    for name, metrics in results.items():
        print(f"{name:36s} {format_metrics(metrics)}")
    print("=" * 65)

    output_csv = args.output_csv
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(BASE_DIR, output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    csv_fields = (
        ["timestamp", "split", "baseline", "median_km", "mean_km"]
        + [f"acc@{t}km" for t in (1, 25, 200, 750, 2500)]
    )
    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_fields)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for name, m in results.items():
            writer.writerow(
                [timestamp, args.split, name, m["median_km"], m["mean_km"]]
                + [m[f"acc@{t}km"] for t in (1, 25, 200, 750, 2500)]
            )
    print(f"Results appended to: {output_csv}")


if __name__ == "__main__":
    main()
