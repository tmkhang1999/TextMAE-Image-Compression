"""
Text-guided refinement, the last stage of TextMAE.

1. BLIP captions every ORIGINAL test image (encoder side). The caption is the only
   extra information sent, and its UTF-8 size is added to the bit rate.
2. The SDXL refiner restores detail in the DECODED image (output of evaluate.py),
   guided by that caption.

    python refine.py -d datasets/kodak_test -r results -o results_refined

Needs the optional packages in requirements.txt (transformers, diffusers) and
downloads the BLIP (~1.9 GB) and SDXL refiner (~6 GB in fp16) weights on first use.
Runs on CUDA, Apple Silicon (MPS) or CPU.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from extras.captioning import DEFAULT_CAPTIONER, Captioner
from extras.refiner import Diffuser
from textmae.data.dataset import collect_images


def parse_args():
    parser = argparse.ArgumentParser(description="Caption-guided refinement of decoded images")
    parser.add_argument("-d", "--dataset", required=True, help="Folder with the original test images")
    parser.add_argument("-r", "--reconstructions", required=True, help="Output folder of evaluate.py")
    parser.add_argument("-o", "--output", default="results_refined")
    parser.add_argument("--input_size", type=int, default=224,
                        help="Resolution the bit rate is measured at (as in evaluate.py)")
    parser.add_argument("--strength", type=float, default=0.3, help="Refiner strength (0 = no change)")
    parser.add_argument("--captioner", default=DEFAULT_CAPTIONER,
                        help="BLIP (default) or BLIP-2 checkpoint, e.g. Salesforce/blip2-opt-2.7b")
    return parser.parse_args()


def main(args):
    originals = collect_images(args.dataset)
    if not originals:
        raise SystemExit(f"No images found in {args.dataset}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    captioner, refiner = Captioner(), Diffuser()
    captioner.prepare_model(args.captioner)
    refiner.prepare_model()

    num_pixels = args.input_size * args.input_size
    report = {}
    for original in originals:
        decoded_path = Path(args.reconstructions) / original.name
        if not decoded_path.exists():
            print(f"Skipped {original.name}: no reconstruction in {args.reconstructions}")
            continue

        caption = captioner.generate_caption(Image.open(original).convert("RGB"))
        refined = refiner.refine_image(caption, Image.open(decoded_path).convert("RGB"), args.strength)
        refined.save(out_dir / original.name)

        caption_bits = len(caption.encode("utf-8")) * 8
        report[original.name] = {"caption": caption, "caption_bpp": caption_bits / num_pixels}
        print(f"{original.name}: '{caption}' (+{caption_bits / num_pixels:.4f} bpp)")

    with open(out_dir / "captions.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main(parse_args())
