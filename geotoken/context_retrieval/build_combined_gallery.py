"""
geotoken/context_retrieval/build_combined_gallery.py — combine train and
val into one gallery, for the final retrain after every model/hyperparameter
decision has been locked in using val as a tuning signal.

Once val's job (guiding hyperparameter choices) is done, folding it back
into training data is a standard, legitimate move — the test set stays
completely untouched throughout, and is still evaluated exactly once at
the end.

Usage (Colab):
  !python geotoken/context_retrieval/build_combined_gallery.py
"""
import os
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_GALLERY_PATH = os.path.join(BASE_DIR, "geotoken", "context_retrieval", "gallery_database.pt")
VAL_FEATURES_PATH = os.path.join(BASE_DIR, "geotoken", "context_retrieval", "val_features.pt")
OUTPUT_PATH = os.path.join(BASE_DIR, "geotoken", "context_retrieval", "gallery_database_train_val.pt")


def main():
  print("=" * 65)
  print("[BUILD COMBINED GALLERY] train + val")
  print("=" * 65)

  assert os.path.exists(TRAIN_GALLERY_PATH), f"Missing {TRAIN_GALLERY_PATH}"
  assert os.path.exists(VAL_FEATURES_PATH), f"Missing {VAL_FEATURES_PATH}"

  train_data = torch.load(TRAIN_GALLERY_PATH, weights_only=True)
  val_data = torch.load(VAL_FEATURES_PATH, weights_only=True)

  train_n = train_data["embeddings"].shape[0]
  val_n = val_data["embeddings"].shape[0]
  print(f"Train gallery : {train_n} images")
  print(f"Val features  : {val_n} images")

  combined = {
      "embeddings": torch.cat([train_data["embeddings"], val_data["embeddings"]], dim=0),
      "s2_tokens": torch.cat([train_data["s2_tokens"], val_data["s2_tokens"]], dim=0),
      "coords": torch.cat([train_data["coords"], val_data["coords"]], dim=0),
  }

  combined_n = combined["embeddings"].shape[0]
  assert combined_n == train_n + val_n, (
      f"Combined count {combined_n} != train {train_n} + val {val_n} — "
      f"something is wrong before saving."
  )

  torch.save(combined, OUTPUT_PATH)
  print(f"Combined gallery: {combined_n} images ({train_n} + {val_n})")
  print(f"Saved to: {OUTPUT_PATH}")
  print("=" * 65)


if __name__ == "__main__":
  main()
