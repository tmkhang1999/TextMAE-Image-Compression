"""Compress and decompress a test folder with trained checkpoints:  bash scripts/evaluate.sh"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import compressai
import torch
from compressai.zoo import load_state_dict
from torch.utils.data import DataLoader

from models import MODELS, build_model
from models.patch_selection import PATCH_SELECTORS
from src.utils.dataset_utils import ImageScoreDataset
from src.utils.metrics import evaluate_model

# Used when neither the command line nor the checkpoint specify a setting
DEFAULTS = {"model": "textmae_base_patch16", "input_size": 224, "num_keep_patches": 144, "patch_selection": "stratified"}


def parse_args():
    p = argparse.ArgumentParser("Evaluate trained TextMAE checkpoints")
    p.add_argument("-d", "--dataset", required=True, help="Test images (scores in <dataset>_scores/test.pt)")
    p.add_argument("-c", "--checkpoint", dest="checkpoints", nargs="+", required=True, help="best_model.pth files")
    p.add_argument("-o", "--output_path", default="results", help="Reconstructions and report.json")
    p.add_argument("--entropy_coder", default=None, help="compressai entropy coder (default: first available)")
    p.add_argument("--entropy_estimation", action="store_true", help="Estimate bpp from likelihoods, no bitstream")
    p.add_argument("--half", action="store_true", help="fp16")
    p.add_argument("--cuda", action="store_true", help="Use the GPU if available")
    # architecture: defaults to what the checkpoint was trained with
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--input_size", type=int)
    p.add_argument("--num_keep_patches", type=int)
    p.add_argument("--patch_selection", choices=sorted(PATCH_SELECTORS))
    return p.parse_args()


def load_model(path, args):
    """Command line > settings stored in the checkpoint > DEFAULTS."""
    ckpt = torch.load(path, map_location="cpu")
    saved = ckpt.get("args", {})
    saved = saved if isinstance(saved, dict) else vars(saved)
    cfg = {k: getattr(args, k) if getattr(args, k) is not None else saved.get(k, d) for k, d in DEFAULTS.items()}
    print(f"Loading {path} as {cfg}")
    model = build_model(cfg["model"], img_size=cfg["input_size"], num_keep_patches=cfg["num_keep_patches"],
                        patch_selection=cfg["patch_selection"])
    model.load_state_dict(load_state_dict(ckpt["model"]))
    return model.eval(), cfg


def main(args):
    coder = args.entropy_coder or compressai.available_entropy_coders()[0]
    compressai.set_entropy_coder(coder)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)  # reproducible timings
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device == "cpu":
        print("WARNING: --cuda requested but CUDA is not available, using CPU.", file=sys.stderr)

    results = defaultdict(list)
    for path in args.checkpoints:
        model, cfg = load_model(path, args)
        model = model.to(device)
        model.update(force=True)
        dataset = ImageScoreDataset("test", args.dataset, cfg["input_size"])
        loader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=device == "cuda")
        metrics = evaluate_model(model, loader, [p.name for p in dataset.img_paths], args.output_path,
                                 args.entropy_estimation, args.half)
        for k, v in metrics.items():
            results[k].append(v)

    report = {"name": "TextMAE", "description": f"Inference ({'entropy estimation' if args.entropy_estimation else coder})",
              "results": results}
    print(json.dumps(report, indent=2))
    (Path(args.output_path) / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())
