"""Optimizers, checkpoint saving / resuming and loading of pretrained MAE weights."""
from pathlib import Path

import torch
import torch.optim as optim

from textmae.utils.distributed import save_on_master

BEST_MODEL_NAME = "best_model.pth"

# Official MAE (facebookresearch/mae) parameter names -> this repository's names.
_MAE_KEY_PREFIXES = {
    "patch_embed.": "encoder_embed.",
    "blocks.": "encoder_blocks.",
    "norm.": "encoder_norm.",
    "cls_token": "cls_token",
    "pos_embed": "encoder_pos_embed",
}


def configure_optimizers(model, args):
    """Return (optimizer, aux_optimizer); the latter only trains the entropy-bottleneck quantiles."""
    params_dict = dict(model.named_parameters())
    aux_names = {n for n, p in params_dict.items() if n.endswith(".quantiles") and p.requires_grad}
    main_names = {n for n, p in params_dict.items() if not n.endswith(".quantiles") and p.requires_grad}
    assert not (main_names & aux_names)

    optimizer = optim.Adam((params_dict[n] for n in sorted(main_names)), lr=args.learning_rate)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_names)), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer


def load_mae_weights(model, path):
    """
    Initialise the matching parts of the model from an official MAE checkpoint.

    Tensors are matched by (renamed) key and shape; everything else keeps its random
    initialisation. Prints what was loaded so a silent mismatch (e.g. a ViT-Large
    checkpoint into the base model) is visible.
    """
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()

    loaded, skipped = {}, []
    for key, tensor in source.items():
        new_key = key
        for old, new in _MAE_KEY_PREFIXES.items():
            if key.startswith(old):
                new_key = new + key[len(old):]
                break
        if new_key in target and target[new_key].shape == tensor.shape:
            loaded[new_key] = tensor
        else:
            skipped.append(key)

    torch.nn.Module.load_state_dict(model, loaded, strict=False)
    print(f"Pretrained MAE weights from {path}: loaded {len(loaded)} tensors, "
          f"skipped {len(skipped)} (missing in model or different shape)")
    if not loaded:
        print("WARNING: no tensors matched. Check that --model matches the checkpoint "
              "(textmae_large_patch16 for the official ViT-Large MAE weights).")


def resume(args, model, optimizer, aux_optimizer):
    """Restore model and optimizer state from `args.resume` (a local path or https URL)."""
    if not args.resume:
        return
    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    print(f"Resumed model from {args.resume}")

    if "optimizer" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed optimizers, continuing at epoch {args.start_epoch}")


def save_best(args, epoch, model, optimizer, aux_optimizer):
    """Write the full training state to <output_dir>/best_model.pth."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_on_master({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }, output_dir / BEST_MODEL_NAME)
