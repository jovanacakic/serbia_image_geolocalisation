"""
g3_adaptation/dataset.py — training dataset for G3's tri-modal geo-alignment,
adding the text modality on top of the image/GPS pairs.

G3's text is NOT image captions. It is a hierarchical place description
obtained by reverse-geocoding the GPS coordinate — their example, from the
paper: (60.37, 6.72) -> "A photo taken from Vestland, Norway", built with
Nominatim across eight levels (neighborhood, city, county, state, region,
country, country code, continent).

OSV5M already ships reverse-geocoded fields, so no geocoding call is needed.
What Serbia's subset actually supports, out of G3's eight levels:

    country      1 unique ("RS")   -> constant, contributes nothing
    region       2 unique          -> weak (4,391 rows missing)
    sub-region  25 unique          -> moderate (4,693 rows missing)
    city       313 unique          -> the real signal (0 missing)

So this is a much flatter hierarchy than G3 gets worldwide, where the
country/continent levels carry most of the discriminative power. Levels that
are missing for a row are simply dropped from its description rather than
filled with a placeholder, so a row never claims a location it lacks.

Note the text is a deterministic *coarse* function of the GPS label, not
independent evidence: its role is hierarchical supervision — an easier,
discretized alignment target alongside the hard continuous one. It is used
only during training; geo-verification at inference is image<->GPS only, so
no text is required (or available) at prediction time.
"""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from geoclip_adaptation.dataset import TRAIN_TRANSFORM, EVAL_TRANSFORM

# "RS" is the raw OSV5M country code; spelled out so the text the encoder
# sees reads like natural language, matching G3's phrasing.
COUNTRY_NAME = "Serbia"


def build_place_text(row):
    """G3-style description, coarsest-informative levels last. `country` is
    constant here, but kept so the string reads like G3's own examples."""
    parts = [row[c] for c in ("city", "sub-region", "region")
             if isinstance(row.get(c), str) and row[c].strip()]
    parts.append(COUNTRY_NAME)
    return "A photo taken from " + ", ".join(parts)


class SerbiaG3Dataset(Dataset):
    """Returns (pixel_values, coords, text_idx). `text_idx` indexes into
    `self.vocab`, so the frozen text tower can be run once per unique string
    instead of once per image — there are only a few hundred distinct
    descriptions across the whole split."""

    def __init__(self, csv_path, raw_images_dir, split_indices_path, train=True):
        df_all = pd.read_csv(csv_path)
        indices = torch.load(split_indices_path, weights_only=True)
        self.df = df_all.iloc[indices.tolist()].reset_index(drop=True)
        self.img_dir = raw_images_dir
        self.transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM

        texts = [build_place_text(self.df.iloc[i]) for i in range(len(self.df))]
        self.vocab = sorted(set(texts))
        lookup = {t: i for i, t in enumerate(self.vocab)}
        self.text_idx = torch.tensor([lookup[t] for t in texts], dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.img_dir, f"{row['id']}.jpg")
        image = (Image.open(path).convert("RGB") if os.path.exists(path)
                 else Image.new("RGB", (224, 224), (0, 0, 0)))
        coords = torch.tensor([row["latitude"], row["longitude"]], dtype=torch.float32)
        return self.transform(image), coords, self.text_idx[i]
