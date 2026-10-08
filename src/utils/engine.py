"""One epoch of training / validation for the rate-distortion objective."""
import torch

LOSS_KEYS = ("loss", "L1_loss", "ssim_loss", "vgg_loss", "bpp_loss")


class AverageMeter:
    """Running average."""

    def __init__(self):
        self.sum = self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def log_line(prefix, meters):
    return prefix + " | ".join(f"{k} {m.avg:.4f}" for k, m in meters.items())


def train_one_epoch(model, criterion, dataloader, optimizer, aux_optimizer, epoch,
                    clip_max_norm=0.0, writer=None, accum_iter=1, print_freq=20):
    """The main optimizer minimises the rate-distortion loss; the auxiliary one the entropy-bottleneck quantiles."""
    model.train()
    device = next(model.parameters()).device
    meters = {k: AverageMeter() for k in (*LOSS_KEYS, "aux_loss")}
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    for step, (samples, _, scores) in enumerate(dataloader):
        samples, scores = samples.to(device, non_blocking=True), scores.to(device, non_blocking=True)
        out = criterion(model(samples, scores), samples)
        aux_loss = model.aux_loss()

        # scale so that the accumulated gradient is the average over accum_iter batches
        (out["loss"] / accum_iter).backward()
        (aux_loss / accum_iter).backward()
        if (step + 1) % accum_iter == 0:
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            aux_optimizer.step()
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()

        values = {k: out[k].item() for k in LOSS_KEYS}
        values["aux_loss"] = aux_loss.item()
        for k, v in values.items():
            meters[k].update(v)
        if writer is not None and (step + 1) % accum_iter == 0:
            global_step = int((step / len(dataloader) + epoch) * 1000)  # keeps curves comparable across batch sizes
            for k, v in {**values, "lr": optimizer.param_groups[0]["lr"]}.items():
                writer.add_scalar(k, v, global_step)
        if step % print_freq == 0:
            print(log_line(f"Epoch {epoch} [{step}/{len(dataloader)}] ", meters))
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(model, criterion, dataloader, epoch=0):
    model.eval()
    device = next(model.parameters()).device
    meters = {k: AverageMeter() for k in (*LOSS_KEYS, "aux_loss")}
    for samples, _, scores in dataloader:
        samples, scores = samples.to(device, non_blocking=True), scores.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = criterion(model(samples, scores), samples)
            values = {k: out[k].item() for k in LOSS_KEYS}
            values["aux_loss"] = model.aux_loss().item()
        for k, v in values.items():
            meters[k].update(v, samples.shape[0])
    print(log_line(f"Val epoch {epoch}: ", meters) + "\n")
    return {k: m.avg for k, m in meters.items()}
