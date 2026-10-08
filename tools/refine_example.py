"""
Run the text-guided stage on the example used in the README figures and time it.

    python tools/refine_example.py

Input: the 0.12 bpp TextMAE reconstruction of Kodak kodim23, cut from assets/2.png
(the trained model of the original experiments is not available anymore).
Writes assets/figures/kodim23_refined.png and assets/figures/kodim23_caption.json.
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extras.captioning import DEFAULT_CAPTIONER, Captioner  # noqa: E402
from extras.refiner import SDXL_REFINER, Diffuser  # noqa: E402

# Bottom-right tile of assets/2.png: the 0.12 bpp / 27.5 dB reconstruction (left, top, right, bottom)
DECODED_TILE = (595, 300, 738, 443)
REFINER_SIZE = 1024  # SDXL works best around 1024 x 1024
INPUT_SIZE = 224     # resolution the bit rate refers to


def local_or_remote(repo_id):
    """Folder of an already downloaded snapshot (it may hold only the fp16 files), else the repo id."""
    from huggingface_hub import try_to_load_from_cache

    index = try_to_load_from_cache(repo_id, "model_index.json")
    if isinstance(index, str):
        return str(Path(index).parent)
    print(f"No local snapshot of {repo_id}; it will be downloaded")
    return repo_id


def decoded_tile():
    return Image.open(ROOT / "assets" / "2.png").convert("RGB").crop(DECODED_TILE)


def main():
    out_dir = ROOT / "assets" / "figures"
    original = Image.open(ROOT / "datasets" / "kodak" / "kodim23.png").convert("RGB")
    timings = {}

    t = time.time()
    captioner = Captioner()
    captioner.prepare_model()
    timings["load_captioner_s"] = time.time() - t

    t = time.time()
    caption = captioner.generate_caption(original)
    timings["caption_s"] = time.time() - t
    print(f"Caption: '{caption}' ({timings['caption_s']:.1f} s on {captioner.device})")
    del captioner

    t = time.time()
    refiner = Diffuser()
    refiner.prepare_model(local_or_remote(SDXL_REFINER))
    timings["load_refiner_s"] = time.time() - t

    decoded = decoded_tile().resize((REFINER_SIZE, REFINER_SIZE), Image.BICUBIC)
    t = time.time()
    refined = refiner.refine_image(caption, decoded, strength=0.3, seed=0)
    timings["refine_s"] = time.time() - t
    print(f"Refined in {timings['refine_s']:.1f} s on {refiner.device}")

    refined.resize((512, 512), Image.LANCZOS).save(out_dir / "kodim23_refined.png")
    caption_bits = len(caption.encode("utf-8")) * 8
    report = {
        "caption": caption,
        "caption_bytes": caption_bits // 8,
        "caption_bpp_at_224": caption_bits / INPUT_SIZE ** 2,
        "captioner": DEFAULT_CAPTIONER,
        "refiner": SDXL_REFINER,
        "strength": 0.3,
        "device": refiner.device,
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    (out_dir / "kodim23_caption.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
