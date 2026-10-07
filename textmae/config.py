"""Command-line options for the three entry points (train.py, evaluate.py, generate_scores.py)."""
import argparse

from textmae.models.patch_selection import PATCH_SELECTORS
from textmae.models.textmae import MODELS


def _add_model_options(parser, defaults=True):
    """Options that describe the architecture; at evaluation time they default to the checkpoint's."""
    d = (lambda value: value) if defaults else (lambda value: None)
    group = parser.add_argument_group("model")
    group.add_argument("--model", default=d("textmae_base_patch16"), choices=sorted(MODELS),
                       help="Architecture preset")
    group.add_argument("--input_size", type=int, default=d(224),
                       help="Images are resized to input_size x input_size")
    group.add_argument("--num_keep_patches", type=int, default=d(144),
                       help="Patches kept by the encoder (a perfect square)")
    group.add_argument("--patch_selection", default=d("stratified"), choices=sorted(PATCH_SELECTORS),
                       help="How the kept patches are chosen from the scores")


def train_parser():
    parser = argparse.ArgumentParser("Train TextMAE for image compression")

    data = parser.add_argument_group("data")
    data.add_argument("-d", "--dataset", required=True,
                      help="Dataset folder with train/ and val/ (scores in <dataset>_scores/)")
    data.add_argument("--num_workers", type=int, default=1)
    data.add_argument("--pin_mem", action=argparse.BooleanOptionalAction, default=True,
                      help="Pin CPU memory in the DataLoader")

    _add_model_options(parser)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("-e", "--epochs", type=int, default=100)
    optim.add_argument("--batch_size", type=int, default=16)
    optim.add_argument("--test_batch_size", type=int, default=8)
    optim.add_argument("--accum_iter", type=int, default=1,
                       help="Gradient accumulation steps (effective batch = batch_size * accum_iter)")
    optim.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    optim.add_argument("--aux_learning_rate", type=float, default=1e-4,
                       help="Learning rate for the entropy-bottleneck quantiles")
    optim.add_argument("--lambda", dest="lmbda", type=float, default=1e-4,
                       help="Rate-distortion trade-off")
    optim.add_argument("--clip_max_norm", type=float, default=1.0,
                       help="Gradient clipping norm, <= 0 disables clipping")

    ckpt = parser.add_argument_group("checkpoints and logging")
    ckpt.add_argument("--pretrained", "--checkpoint", dest="pretrained", default="",
                      help="Official MAE checkpoint used to initialise the matching weights")
    ckpt.add_argument("--resume", default="", help="Resume training from a checkpoint")
    ckpt.add_argument("--start_epoch", type=int, default=0)
    ckpt.add_argument("--output_dir", default="", help="Where to save best_model.pth (empty: no saving)")
    ckpt.add_argument("--log_dir", default="", help="TensorBoard log directory (empty: no logging)")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="cuda", help="cuda or cpu")
    runtime.add_argument("--cuda", action="store_true", help="Deprecated, --device cuda is the default")
    runtime.add_argument("--seed", type=int, default=0)
    return parser


def evaluate_parser():
    parser = argparse.ArgumentParser("Evaluate a trained TextMAE model")
    parser.add_argument("-d", "--dataset", required=True,
                        help="Test image folder (scores in <dataset>_scores/test.pt)")
    parser.add_argument("-c", "--checkpoint", dest="checkpoints", nargs="+", required=True,
                        help="One or more best_model.pth files")
    parser.add_argument("-o", "--output_path", default="results",
                        help="Where reconstructions and report.json are written")
    parser.add_argument("--entropy_coder", default=None,
                        help="compressai entropy coder (default: first available)")
    parser.add_argument("--entropy_estimation", action="store_true",
                        help="Estimate bpp from likelihoods instead of coding a real bitstream")
    parser.add_argument("--half", action="store_true", help="Run the model in fp16")
    parser.add_argument("--cuda", action="store_true", help="Use the GPU if available")
    parser.add_argument("-v", "--verbose", action="store_true")
    _add_model_options(parser, defaults=False)
    return parser


def generate_scores_parser():
    parser = argparse.ArgumentParser("Precompute per-patch importance scores")
    parser.add_argument("--training_path", required=True, help="Folder with train/ and val/ images")
    parser.add_argument("--testing_path", required=True, help="Folder with test images")
    parser.add_argument("--input_size", type=int, default=224,
                        help="Must match the --input_size used for training / evaluation")
    parser.add_argument("--patch_size", type=int, default=16)
    return parser
