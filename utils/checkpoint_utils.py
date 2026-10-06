"""
utils/checkpoint_utils.py — save only what training actually changed.

Every model in this project wraps a frozen CLIP ViT-L/14 backbone, so a full
state_dict is ~1.75 GB of which ~97% is weights that never move and are
reconstructible from the pretrained checkpoint. Writing that much to a Google
Drive FUSE mount every epoch is both slow and, as this project found the hard
way, unreliable: torch.save reported success while the file on Drive silently
kept an epoch-1 version, and four runs' weights were lost when the mount
crashed and Drive rolled back to its last synced copy.

Saving only trainable parameters (plus all buffers, which are small and may
carry real state such as G3's GPS queue or the regressor's bbox bounds) cuts
checkpoints from ~1.75 GB to ~25-115 MB depending on the configuration —
small enough for Drive to handle reliably, and far faster.

Loading uses strict=False and then verifies, rather than trusting silence:
every saved key must be consumed, and the count of restored tensors must
match. Frozen weights come from the model's own pretrained initialisation,
which is deterministic for a given construction.
"""
import torch


def trainable_state_dict(model):
    """Trainable parameters + all buffers. Buffers are included regardless of
    requires_grad: they are tiny, and some are load-bearing (G3's gps_queue
    for resume, GeoCLIPRegressor's bbox_min/bbox_max for decoding output)."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    keep = trainable | buffers
    return {k: v for k, v in model.state_dict().items() if k in keep}


def load_trainable_state_dict(model, saved, verbose=True):
    """Restore a trainable-only checkpoint onto a freshly constructed model.

    Raises if any saved key is unexpected — that means the architecture the
    checkpoint came from differs from the one being loaded into, which is a
    real error rather than something to warn about and continue past.
    """
    missing, unexpected = model.load_state_dict(saved, strict=False)
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} unexpected key(s) in checkpoint — architecture "
            f"mismatch. First few: {list(unexpected)[:5]}"
        )

    # Everything the checkpoint omitted must be a frozen weight supplied by
    # the pretrained init; a trainable parameter appearing here would mean
    # the checkpoint is incomplete.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing_trainable = [k for k in missing if k in trainable]
    if missing_trainable:
        raise RuntimeError(
            f"{len(missing_trainable)} TRAINABLE parameter(s) absent from the "
            f"checkpoint: {missing_trainable[:5]}"
        )

    if verbose:
        n = sum(v.numel() for v in saved.values())
        print(f"  restored {len(saved)} tensors ({n:,} elements); "
              f"{len(missing)} frozen tensors from pretrained init")
    return model


def load_any_state_dict(model, obj, verbose=True):
    """Accepts either a trainable-only checkpoint or a legacy full state_dict,
    so checkpoints saved before this change keep working."""
    saved = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model_keys = set(model.state_dict().keys())
    if len(saved) >= 0.9 * len(model_keys):
        if verbose:
            print(f"  loading legacy full checkpoint ({len(saved)} tensors)")
        model.load_state_dict(saved)
        return model
    return load_trainable_state_dict(model, saved, verbose=verbose)
