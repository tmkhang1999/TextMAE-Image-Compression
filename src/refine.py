"""Text-guided refinement, the last stage of TextMAE:  bash scripts/refine.sh

BLIP captions every ORIGINAL test image (encoder side; the caption is the only extra information sent and its
UTF-8 size is added to the bit rate), then the SDXL refiner restores detail in the DECODED image guided by it.
Needs transformers and diffusers; downloads about 8 GB of weights on first use. Runs on CUDA, Apple MPS or CPU.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from models.text_guidance import CAPTIONER, Captioner, Refiner
from src.utils.dataset_utils import collect_images


def parse_args():
    p = argparse.ArgumentParser("Caption-guided refinement of decoded images")
    p.add_argument("-d", "--dataset", required=True, help="Original test images")
    p.add_argument("-r", "--reconstructions", required=True, help="Output folder of src.inference")
    p.add_argument("-o", "--output", default="results_refined")
    p.add_argument("--input_size", type=int, default=224, help="Resolution the bit rate is measured at")
    p.add_argument("--strength", type=float, default=0.3, help="0 = no change")
    p.add_argument("--captioner", default=CAPTIONER, help="BLIP (default) or BLIP-2, e.g. Salesforce/blip2-opt-2.7b")
    return p.parse_args()


def main(args):
    originals = collect_images(args.dataset)
    if not originals:
        raise SystemExit(f"No images found in {args.dataset}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    caption_of, refine = Captioner(args.captioner), Refiner()

    report = {}
    for original in originals:
        decoded = Path(args.reconstructions) / original.name
        if not decoded.exists():
            print(f"Skipped {original.name}: no reconstruction in {args.reconstructions}")
            continue
        caption = caption_of(Image.open(original).convert("RGB"))
        refine(caption, Image.open(decoded).convert("RGB"), args.strength).save(out_dir / original.name)
        bpp = len(caption.encode("utf-8")) * 8 / args.input_size ** 2
        report[original.name] = {"caption": caption, "caption_bpp": bpp}
        print(f"{original.name}: '{caption}' (+{bpp:.4f} bpp)")
    (out_dir / "captions.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())
