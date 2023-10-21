import torch
import logging
from diffusers import StableDiffusionXLImg2ImgPipeline

log = logging.getLogger(__name__)


class Diffusion:
    def __init__(self, device="cuda"):
        self.device = device

    def prepare_model(self, model_name="stable-diffusion-xl-refiner-1.0"):
        if model_name == "stable-diffusion-xl-refiner-1.0":
            return stable_diffusion_xl_refiner_1(self.device)
        else:
            log.error(f"Model name '{model_name}' is not recognized.")


def stable_diffusion_xl_refiner_1(device):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    return pipe
