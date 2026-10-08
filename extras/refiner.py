"""Stable Diffusion XL refiner: sharpen a decoded image, guided by its caption."""
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

from extras.device import best_device

SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"


class Diffuser:
    def __init__(self):
        self.model = None
        self.device, self.dtype = best_device()

    def prepare_model(self, model_name=SDXL_REFINER):
        """`model_name` is a Hugging Face repo id or a local folder with the pipeline files."""
        # The fp16 weight files are half the download; fall back to the full ones if absent
        variant = "fp16" if self.dtype == torch.float16 else None
        try:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_name, torch_dtype=self.dtype, variant=variant)
        except (OSError, ValueError) as err:
            print(f"fp16 weights not available ({err}); loading the default weights")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model = pipe.to(self.device)

    def refine_image(self, caption, image, strength=0.3, seed=0):
        """
        Args:
            caption (str): Text prompt, e.g. the BLIP caption of the original image.
            image (PIL.Image): Decoded image.
            strength (float): How far the refiner may move away from `image` (0 = unchanged).
        """
        if self.model is None:
            raise RuntimeError("Call prepare_model() before refine_image()")
        generator = torch.Generator("cpu").manual_seed(seed)
        return self.model(caption, image=image, strength=strength, generator=generator).images[0]
