"""One epoch of training / validation for the rate-distortion objective."""
import torch

from textmae.utils import distributed, logger

# Quantities that are tracked, printed and written to TensorBoard.
LOSS_KEYS = ("loss", "L1_loss", "ssim_loss", "vgg_loss", "bpp_loss")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(model, criterion, dataloader, optimizer, aux_optimizer, epoch,
                    clip_max_norm=0.0, writer=None, accum_iter=1, print_freq=20):
    """
    Train for one epoch.

    The main optimizer minimises the rate-distortion loss; the auxiliary optimizer
    updates the entropy-bottleneck quantiles with the model's aux loss.

    Args:
        model: TextMAE model.
        criterion: RateDistortionLoss.
        dataloader: Yields (images, original sizes, patch scores).
        optimizer, aux_optimizer: Main and auxiliary optimizers.
        epoch (int): Current epoch, used for logging.
        clip_max_norm (float): Gradient clipping threshold, <= 0 disables clipping.
        writer: Optional TensorBoard SummaryWriter.
        accum_iter (int): Gradient accumulation steps.

    Returns:
        dict: Epoch averages of the tracked losses and learning rates.
    """
    model.train()
    device = next(model.parameters()).device

    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", logger.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    for step, (samples, _, total_scores) in enumerate(
            metric_logger.log_every(dataloader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        total_scores = total_scores.to(device, non_blocking=True)

        out_net = model(samples, total_scores)
        out_criterion = criterion(out_net, samples)
        aux_loss = model.aux_loss()

        # Scale so that the accumulated gradient is the average over accum_iter batches.
        (out_criterion["loss"] / accum_iter).backward()
        (aux_loss / accum_iter).backward()

        if (step + 1) % accum_iter == 0:
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            aux_optimizer.step()
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

        _sync(device)

        values = {key: out_criterion[key].item() for key in LOSS_KEYS}
        values["aux_loss"] = aux_loss.item()
        values["lr"] = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(**values)

        # All ranks must take part in the reduction, only rank 0 has a writer.
        reduced = {key: distributed.all_reduce_mean(value) for key, value in values.items()}
        if writer is not None and (step + 1) % accum_iter == 0:
            # Epoch x1000 keeps curves comparable when the batch size changes.
            global_step = int((step / len(dataloader) + epoch) * 1000)
            for key, value in reduced.items():
                writer.add_scalar(key, value, global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, criterion, dataloader, epoch=0, print_freq=10):
    """
    Evaluate the training objective on a validation set.

    Returns:
        dict: Averages of the tracked losses (plus "aux_loss").
    """
    model.eval()
    device = next(model.parameters()).device

    metric_logger = logger.MetricLogger(delimiter="  ")
    use_amp = device.type == "cuda"

    for samples, _, total_scores in metric_logger.log_every(dataloader, print_freq, "Val:"):
        samples = samples.to(device, non_blocking=True)
        total_scores = total_scores.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out_net = model(samples, total_scores)
            out_criterion = criterion(out_net, samples)
            aux_loss = model.aux_loss()

        values = {key: out_criterion[key].item() for key in LOSS_KEYS}
        values["aux_loss"] = aux_loss.item()
        metric_logger.update(**values)

    metric_logger.synchronize_between_processes()
    stats = {k: round(meter.global_avg, 4) for k, meter in metric_logger.meters.items()}
    print(f"Val epoch {epoch}: " + " | ".join(f"{k} {v:.4f}" for k, v in stats.items()) + "\n")
    return stats
