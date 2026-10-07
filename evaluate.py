"""Evaluate trained checkpoints on a test folder. Example: see test.sh."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import compressai
import torch
from compressai.zoo import load_state_dict
from torch.utils.data import DataLoader

from textmae.config import evaluate_parser
from textmae.data import ImageScoreDataset
from textmae.engine.evaluate import evaluate_model
from textmae.models import build_model

# Used when neither the command line nor the checkpoint specify a setting.
MODEL_DEFAULTS = {
    "model": "textmae_base_patch16",
    "input_size": 224,
    "num_keep_patches": 144,
    "patch_selection": "stratified",
}


def resolve_model_config(args, checkpoint):
    """Command line > settings stored in the checkpoint > defaults."""
    saved = checkpoint.get("args", {})
    saved = vars(saved) if not isinstance(saved, dict) else saved
    config = {}
    for key, default in MODEL_DEFAULTS.items():
        value = getattr(args, key)
        config[key] = value if value is not None else saved.get(key, default)
    return config


def load_model(path, args):
    checkpoint = torch.load(path, map_location="cpu")
    config = resolve_model_config(args, checkpoint)
    print(f"Loading {path} as {config}")

    model = build_model(
        config["model"], img_size=config["input_size"],
        num_keep_patches=config["num_keep_patches"], patch_selection=config["patch_selection"])
    model.load_state_dict(load_state_dict(checkpoint["model"]))
    model.eval()
    return model, config


def main(argv):
    args = evaluate_parser().parse_args(argv)

    coder = args.entropy_coder or compressai.available_entropy_coders()[0]
    compressai.set_entropy_coder(coder)

    # Fixed seed-independent behaviour for reproducible timings
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and device == "cpu":
        print("WARNING: --cuda requested but CUDA is not available, using CPU.", file=sys.stderr)

    results = defaultdict(list)
    for checkpoint_path in args.checkpoints:
        if args.verbose:
            print(f"Evaluating {checkpoint_path}", file=sys.stderr)

        model, config = load_model(checkpoint_path, args)
        model = model.to(device)
        model.update(force=True)

        dataset = ImageScoreDataset("test", args.dataset, input_size=config["input_size"])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                pin_memory=device == "cuda")
        file_names = [p.name for p in dataset.img_paths]

        metrics = evaluate_model(
            model, dataloader, file_names, args.output_path,
            entropy_estimation=args.entropy_estimation, half=args.half)
        for key, value in metrics.items():
            results[key].append(value)

    report = {
        "name": "TextMAE",
        "description": f"Inference ({'entropy estimation' if args.entropy_estimation else coder})",
        "results": results,
    }
    print(json.dumps(report, indent=2))
    with open(Path(args.output_path) / "report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
