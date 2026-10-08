"""Rebuild every figure of the README:  python -m tools.figures.make_figures

Writes to assets/figures/: overview.svg, codec.svg, patch_selection.png, rd_kodak.png and, if
tools/figures/refine_example.py was run, refinement.jpg.

TextMAE numbers and the decoded example come from assets/1.png and assets/2.png (outputs of the trained model of the
original experiments). The refined example and its caption come from refine_example.py. JPEG / WebP are computed here
with Pillow on the same 224 x 224 inputs.
"""
from tools.figures.common import ROOT, kept_mask, load_inputs, masked_view, decoded_rgb, refined_example, score_heatmap
from tools.figures.make_diagrams import make_codec_svg, make_overview_svg
from tools.figures.make_plots import make_patch_figure, make_rd_figure, make_refinement_figure


def main():
    out_dir = ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb, s_map, t_map, scores = load_inputs(ROOT / "datasets" / "kodak" / "kodim23.png")
    make_patch_figure(rgb, s_map, t_map, scores, out_dir / "patch_selection.png")
    make_rd_figure(out_dir / "rd_kodak.png")

    refined, meta = refined_example()
    example = {
        "input": rgb,
        "score": score_heatmap(scores),
        "kept": masked_view(rgb, kept_mask(scores, "stratified")),
        "decoded": decoded_rgb(),
        "refined": refined,
        "caption": meta["caption"] if meta else "caption",
        "caption_bytes": meta["caption_bytes"] if meta else "a few",
        "caption_bpp": meta["caption_bpp_at_224"] if meta else 0.0,
        # total rate of the refined image: decoded image (0.12 bpp) + caption
        "refined_bpp": f"{0.12 + meta['caption_bpp_at_224']:.2f}" if meta else "",
    }
    make_overview_svg(example, out_dir / "overview.svg")
    make_codec_svg(example, out_dir / "codec.svg")
    if refined is not None:
        make_refinement_figure(refined, meta, out_dir / "refinement.jpg")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
