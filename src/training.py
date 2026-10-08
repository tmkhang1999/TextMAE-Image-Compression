"""Train TextMAE:  bash scripts/train.sh  (or python -m src.training -d datasets/<name> --output_dir weights)."""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MODELS, build_model
from models.patch_selection import PATCH_SELECTORS
from src.losses import RateDistortionLoss
from src.utils.checkpoint_utils import load_mae_weights, resume, save_best
from src.utils.dataset_utils import ImageScoreDataset
from src.utils.engine import train_one_epoch, validate
from src.utils.optimizers import configure_optimizers


def parse_args():
    p = argparse.ArgumentParser("Train TextMAE for image compression")
    p.add_argument("-d", "--dataset", required=True, help="Folder with train/ and val/ (scores in <dataset>_scores/)")
    p.add_argument("--model", default="textmae_base_patch16", choices=sorted(MODELS))
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--num_keep_patches", type=int, default=144, help="Kept patches (a perfect square)")
    p.add_argument("--patch_selection", default="stratified", choices=sorted(PATCH_SELECTORS))
    p.add_argument("-e", "--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=8)
    p.add_argument("--accum_iter", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    p.add_argument("--aux_learning_rate", type=float, default=1e-4, help="Entropy-bottleneck quantiles")
    p.add_argument("--lambda", dest="lmbda", type=float, default=1e-4, help="Rate-distortion trade-off")
    p.add_argument("--clip_max_norm", type=float, default=1.0, help="<= 0 disables gradient clipping")
    p.add_argument("--pretrained", default="", help="Official MAE checkpoint for the matching weights")
    p.add_argument("--resume", default="", help="Continue from a best_model.pth")
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--output_dir", default="", help="Where best_model.pth is saved (empty: not saved)")
    p.add_argument("--log_dir", default="", help="TensorBoard logs (empty: none)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (falls back to cpu)")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main(args):
    print(str(args).replace(", ", ",\n"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != args.device:
        print(f"WARNING: '{args.device}' is not available, training on {device}.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_set = ImageScoreDataset("train", args.dataset, args.input_size)
    val_set = ImageScoreDataset("val", args.dataset, args.input_size)
    loader = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, **loader)
    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    model = build_model(args.model, img_size=args.input_size, num_keep_patches=args.num_keep_patches,
                        patch_selection=args.patch_selection)
    if args.pretrained:
        load_mae_weights(model, args.pretrained)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    optimizer, aux_optimizer = configure_optimizers(model, args)
    resume(args, model, optimizer, aux_optimizer)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    best = float("inf")
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, epoch,
                        args.clip_max_norm, writer, args.accum_iter)
        stats = validate(model, criterion, val_loader, epoch)
        if args.output_dir and stats["loss"] < best:
            best = stats["loss"]
            save_best(args, epoch, model, optimizer, aux_optimizer)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main(parse_args())
