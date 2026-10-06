"""
error_analysis.py — diagnose WHERE and WHY a model fails, beyond the single
aggregate median that the results table reports.

Aggregate error hides everything interesting: whether mistakes are uniformly
spread or concentrated in one region, whether failures correlate with terrain
or scene type, and whether a model's error distribution has a heavy tail of
catastrophic misses or is uniformly mediocre. Those distinctions matter for
the report's discussion far more than a single number, and they are also how
you decide what to fix next.

Groups by the metadata OSV5M already ships — region, sub-region,
land_cover, climate, city — rather than hand-defined latitude bands.

Runs on VAL by default and should stay there: this is a development
diagnostic, and inspecting per-image test errors would turn the one-shot test
measurement into an iterative tuning signal.

Produces:
  - console summary: percentiles, per-group error tables, worst predictions
  - <out_dir>/<label>_per_image.csv   every image, its error, its metadata
  - <out_dir>/<label>_map.html        folium map, true locations coloured by
                                      error, with lines to the predictions
                                      for the worst cases

Usage (Colab, after %cd into the project root):
  !python error_analysis.py --model g3 \
      --checkpoint g3_adaptation/checkpoints/g3_text/best_model.pt

  !python error_analysis.py --model geoclip \
      --checkpoint geoclip_adaptation/checkpoints/v1_full_unfreeze/best_model.pt \
      --adaptation full_unfreeze
"""
import os
import sys
import argparse

import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from evaluate_test import (
    predict_geoclip, predict_g3, predict_geotoken, predict_centroid,
    resolve, CSV_PATH, SPLITS_DIR,
)
from utils.eval_utils import haversine_distance

# Metadata columns worth grouping by. Skipped automatically when a column is
# absent, constant (e.g. `country` is always "RS" here), or too sparse to say
# anything.
GROUP_COLUMNS = ["region", "sub-region", "land_cover", "climate", "drive_side"]
MIN_GROUP_SIZE = 20


def load_split_metadata(split):
    df_all = pd.read_csv(CSV_PATH)
    idx = torch.load(os.path.join(SPLITS_DIR, f"{split}_indices.pt"), weights_only=True)
    return df_all.iloc[idx.tolist()].reset_index(drop=True)


def group_table(df, col, min_size=MIN_GROUP_SIZE):
    if col not in df.columns or df[col].nunique(dropna=True) < 2:
        return None
    grouped = df.groupby(col)["error_km"].agg(["count", "median", "mean"])
    grouped = grouped[grouped["count"] >= min_size].sort_values("median")
    return grouped if len(grouped) >= 2 else None


def build_map(df, out_path, worst_n=25):
    try:
        import folium
    except ImportError:
        print("[skip] folium not installed — no map written.")
        return

    m = folium.Map(location=[44.0, 20.9], zoom_start=7, tiles="cartodbpositron")

    # Error buckets, coarse enough to read at a glance.
    def colour(e):
        return ("#2166ac" if e <= 1 else "#67a9cf" if e <= 25 else
                "#fddbc7" if e <= 100 else "#ef8a62" if e <= 200 else "#b2182b")

    truth = folium.FeatureGroup(name="Ground truth (coloured by error)")
    for _, r in df.iterrows():
        folium.CircleMarker(
            [r["latitude"], r["longitude"]], radius=2.5,
            color=colour(r["error_km"]), fill=True, fill_opacity=0.7, weight=0,
            popup=f"id {r['id']}<br>{r['error_km']:.1f} km<br>{r.get('city', '')}",
        ).add_to(truth)
    truth.add_to(m)

    # The worst cases get an explicit line to where the model actually put
    # them — direction and distance of failure is often more informative than
    # the magnitude alone (e.g. everything pulled toward one city).
    worst = df.nlargest(worst_n, "error_km")
    fg = folium.FeatureGroup(name=f"Worst {worst_n}: true -> predicted")
    for _, r in worst.iterrows():
        folium.PolyLine(
            [[r["latitude"], r["longitude"]], [r["pred_lat"], r["pred_lon"]]],
            color="#b2182b", weight=1.2, opacity=0.8,
        ).add_to(fg)
        folium.CircleMarker(
            [r["pred_lat"], r["pred_lon"]], radius=3, color="#000000",
            fill=True, fill_opacity=0.9, weight=0.5,
            popup=f"PREDICTED for id {r['id']}<br>{r['error_km']:.1f} km",
        ).add_to(fg)
    fg.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_path)
    print(f"Map written to: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["geotoken", "geoclip", "g3", "centroid"], required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="error_analysis")
    p.add_argument("--worst_n", type=int, default=15)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--adaptation", choices=["lora", "full_unfreeze"], default="lora")
    p.add_argument("--unfreeze_last_n_layers", type=int, default=2)
    p.add_argument("--image_adaptation", choices=["frozen_mlp", "lora"], default="frozen_mlp")
    p.add_argument("--lora_targets", type=str, default="q,k,v,out")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--gallery_path", type=str,
                   default="geotoken/context_retrieval/gallery_database.pt")
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument("--clip_dim", type=int, default=768,
                   help="768 for raw CLIP features, 1280 for geo-aligned.")
    p.add_argument("--features_path", type=str, default=None,
                   help="Override the geotoken feature file for this split.")
    p.add_argument("--num_encoder_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()

    if args.model != "centroid" and not args.checkpoint:
        p.error(f"--checkpoint is required for --model {args.model}")

    label = args.label or (
        f"{args.model}"
        + (f"_{os.path.basename(os.path.dirname(resolve(args.checkpoint)))}" if args.checkpoint else "")
        + ("_tta" if args.tta else ""))

    print("=" * 72)
    print(f"ERROR ANALYSIS — {label} ({args.split})")
    print("=" * 72)
    if args.split == "test":
        print("[warning] Running on TEST. This is a development diagnostic; "
              "inspecting per-image test errors turns the one-shot test "
              "measurement into a tuning signal. Use --split val unless you "
              "have a specific reason.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = {"geoclip": predict_geoclip, "g3": predict_g3,
                 "geotoken": predict_geotoken, "centroid": predict_centroid}[args.model]
    pred_coords, true_coords = predictor(args, device)
    errors = haversine_distance(pred_coords, true_coords)

    # Predictors iterate the split in order with shuffle=False, so row i of
    # the metadata corresponds to prediction i.
    df = load_split_metadata(args.split)
    assert len(df) == len(errors), f"metadata {len(df)} != predictions {len(errors)}"
    df["pred_lat"] = pred_coords[:, 0].numpy()
    df["pred_lon"] = pred_coords[:, 1].numpy()
    df["error_km"] = errors.numpy()

    print(f"n = {len(df)}   median = {errors.median():.2f} km   mean = {errors.mean():.2f} km")
    print("\nError percentiles (km):")
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print("  " + "  ".join(f"p{int(q*100)}={torch.quantile(errors, q):.1f}" for q in qs))
    print(f"  max={errors.max():.1f}")

    print("\nShare of predictions by error band:")
    bands = [(0, 1), (1, 25), (25, 100), (100, 200), (200, float("inf"))]
    for lo, hi in bands:
        share = ((errors > lo) & (errors <= hi)).float().mean().item() * 100
        hi_s = "inf" if hi == float("inf") else f"{hi:g}"
        print(f"  ({lo:>3g}, {hi_s:>3}] km : {share:5.2f}%")

    for col in GROUP_COLUMNS:
        table = group_table(df, col)
        if table is None:
            continue
        print(f"\nError by {col} (groups with >= {MIN_GROUP_SIZE} images, best first):")
        print(f"  {'value':<34} {'n':>5} {'median':>8} {'mean':>8}")
        for value, row in table.iterrows():
            print(f"  {str(value)[:34]:<34} {int(row['count']):>5} "
                  f"{row['median']:>8.1f} {row['mean']:>8.1f}")

    top_cities = df["city"].value_counts()
    top_cities = top_cities[top_cities >= MIN_GROUP_SIZE].index[:10]
    if len(top_cities) >= 2:
        sub = df[df["city"].isin(top_cities)].groupby("city")["error_km"].agg(
            ["count", "median"]).sort_values("median")
        print(f"\nError by city (10 most frequent, best first):")
        print(f"  {'city':<34} {'n':>5} {'median':>8}")
        for value, row in sub.iterrows():
            print(f"  {str(value)[:34]:<34} {int(row['count']):>5} {row['median']:>8.1f}")

    print(f"\nWorst {args.worst_n} predictions:")
    print(f"  {'id':<20} {'err_km':>8}  {'true (lat,lon)':<22} {'pred (lat,lon)':<22} city")
    for _, r in df.nlargest(args.worst_n, "error_km").iterrows():
        print(f"  {str(r['id'])[:20]:<20} {r['error_km']:>8.1f}  "
              f"({r['latitude']:.3f}, {r['longitude']:.3f})".ljust(56)
              + f"({r['pred_lat']:.3f}, {r['pred_lon']:.3f})".ljust(23)
              + str(r.get("city", ""))[:20])

    print(f"\nBest {args.worst_n} predictions:")
    for _, r in df.nsmallest(args.worst_n, "error_km").iterrows():
        print(f"  {str(r['id'])[:20]:<20} {r['error_km']:>8.2f}  {str(r.get('city',''))[:30]}")

    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    keep = [c for c in ["id", "latitude", "longitude", "pred_lat", "pred_lon", "error_km",
                        "region", "sub-region", "city", "land_cover", "climate",
                        "drive_side", "sequence"] if c in df.columns]
    csv_out = os.path.join(out_dir, f"{label}_per_image.csv")
    df[keep].sort_values("error_km", ascending=False).to_csv(csv_out, index=False)
    print(f"\nPer-image errors written to: {csv_out}")

    build_map(df, os.path.join(out_dir, f"{label}_map.html"), worst_n=25)
    print("=" * 72)


if __name__ == "__main__":
    main()
