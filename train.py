"""Train TextMAE. Example: see train.sh."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from textmae.config import train_parser
from textmae.data import ImageScoreDataset
from textmae.engine.train import train_one_epoch, validate
from textmae.losses import RateDistortionLoss
from textmae.models import build_model
from textmae.utils import checkpoint, distributed


def main(args):
    print(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    print(str(args).replace(", ", ",\n"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != args.device:
        print(f"WARNING: '{args.device}' is not available, training on {device}.")

    seed = args.seed + distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Data
    train_dataset = ImageScoreDataset("train", args.dataset, input_size=args.input_size)
    val_dataset = ImageScoreDataset("val", args.dataset, input_size=args.input_size)
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=distributed.get_world_size(),
        rank=distributed.get_rank(), shuffle=True)
    train_loader = DataLoader(
        train_dataset, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    writer = None
    if distributed.is_main_process() and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)

    # Model
    model = build_model(
        args.model, img_size=args.input_size, num_keep_patches=args.num_keep_patches,
        patch_selection=args.patch_selection)
    if args.pretrained:
        checkpoint.load_mae_weights(model, args.pretrained)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    optimizer, aux_optimizer = checkpoint.configure_optimizers(model, args)
    checkpoint.resume(args, model, optimizer, aux_optimizer)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    best_loss = float("inf")
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, epoch,
                        clip_max_norm=args.clip_max_norm, writer=writer, accum_iter=args.accum_iter)
        stats = validate(model, criterion, val_loader, epoch)

        if args.output_dir and stats["loss"] < best_loss:
            best_loss = stats["loss"]
            checkpoint.save_best(args, epoch, model, optimizer, aux_optimizer)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main(train_parser().parse_args())
