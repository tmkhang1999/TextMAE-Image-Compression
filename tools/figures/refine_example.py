"""Run the text-guided stage on the README example and time it:  python -m tools.figures.refine_example

Input: the 0.12 bpp TextMAE reconstruction of Kodak kodim23, cut from assets/2.png (the trained model of the
original experiments is not available anymore). Writes assets/figures/kodim23_refined.png and kodim23_caption.json.
"""
import json
import time
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from PIL import Image

from models.text_guidance import CAPTIONER, SDXL_REFINER, Captioner, Refiner
from tools.figures.common import DECODED_TILE, INPUT_SIZE, ROOT

REFINER_SIZE = 1024  # SDXL works best around 1024 x 1024


def local_or_remote(repo_id):
    """Folder of an already downloaded snapshot (it may hold only the fp16 files), else the repo id."""
    index = try_to_load_from_cache(repo_id, "model_index.json")
    return str(Path(index).parent) if isinstance(index, str) else repo_id


def main():
    out_dir = ROOT / "assets" / "figures"
    original = Image.open(ROOT / "datasets" / "kodak" / "kodim23.png").convert("RGB")
    timings = {}

    t = time.time()
    caption_of = Captioner()
    timings["load_captioner_s"] = time.time() - t
    t = time.time()
    caption = caption_of(original)
    timings["caption_s"] = time.time() - t
    print(f"Caption: '{caption}' ({timings['caption_s']:.1f} s on {caption_of.device})")
    del caption_of

    t = time.time()
    refine = Refiner(local_or_remote(SDXL_REFINER))
    timings["load_refiner_s"] = time.time() - t
    decoded = Image.open(ROOT / "assets" / "2.png").convert("RGB").crop(DECODED_TILE).resize(
        (REFINER_SIZE, REFINER_SIZE), Image.BICUBIC)
    t = time.time()
    refined = refine(caption, decoded, strength=0.3, seed=0)
    timings["refine_s"] = time.time() - t
    print(f"Refined in {timings['refine_s']:.1f} s on {refine.device}")

    refined.resize((512, 512), Image.LANCZOS).save(out_dir / "kodim23_refined.png")
    bits = len(caption.encode("utf-8")) * 8
    report = {"caption": caption, "caption_bytes": bits // 8, "caption_bpp_at_224": bits / INPUT_SIZE ** 2,
              "captioner": CAPTIONER, "refiner": SDXL_REFINER, "strength": 0.3, "device": refine.device,
              "timings_s": {k: round(v, 1) for k, v in timings.items()}}
    (out_dir / "kodim23_caption.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
