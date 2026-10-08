from pathlib import Path

import torch

BEST_MODEL_NAME = "best_model.pth"

# Official MAE (facebookresearch/mae) parameter names -> this repository's names
_MAE_KEYS = {"patch_embed.": "encoder_embed.", "blocks.": "encoder_blocks.", "norm.": "encoder_norm.",
             "cls_token": "cls_token", "pos_embed": "encoder_pos_embed"}


def load_mae_weights(model, path):
    """Initialise the matching parts of the model from an official MAE checkpoint (by renamed key and shape)
    and print how much was loaded, so a silent mismatch (e.g. ViT-L weights into the base model) is visible."""
    source = torch.load(path, map_location="cpu")
    source = source.get("model", source)
    target = model.state_dict()
    loaded, skipped = {}, 0
    for key, tensor in source.items():
        new = next((v + key[len(k):] for k, v in _MAE_KEYS.items() if key.startswith(k)), key)
        if new in target and target[new].shape == tensor.shape:
            loaded[new] = tensor
        else:
            skipped += 1
    torch.nn.Module.load_state_dict(model, loaded, strict=False)
    print(f"Pretrained MAE weights from {path}: loaded {len(loaded)} tensors, skipped {skipped}")
    if not loaded:
        print("WARNING: no tensors matched; --model must match the checkpoint "
              "(textmae_large_patch16 for the official ViT-Large weights).")


def resume(args, model, optimizer, aux_optimizer):
    """Restore model and optimizers from `args.resume` (local path or https URL)."""
    if not args.resume:
        return
    if args.resume.startswith("https"):
        ckpt = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
    else:
        ckpt = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt and "epoch" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        args.start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from {args.resume}, continuing at epoch {args.start_epoch}")


def save_best(args, epoch, model, optimizer, aux_optimizer):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(), "epoch": epoch, "args": vars(args)},
               Path(args.output_dir) / BEST_MODEL_NAME)
