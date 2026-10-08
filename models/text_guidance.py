"""Text guidance: BLIP / BLIP-2 captions the input, the SDXL refiner restores detail in the decoded image."""
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration, BlipProcessor

CAPTIONER = "Salesforce/blip-image-captioning-large"  # ~1.9 GB; BLIP-2 (blip2-opt-2.7b) needs ~15 GB
SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"


def best_device():
    """CUDA, then Apple Silicon (MPS), then CPU; fp16 on GPUs, fp32 on CPU."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class Captioner:
    def __init__(self, model_name=CAPTIONER):
        self.device, dtype = best_device()
        blip2 = "blip2" in model_name.lower()
        self.processor = (AutoProcessor if blip2 else BlipProcessor).from_pretrained(model_name)
        model_cls = Blip2ForConditionalGeneration if blip2 else BlipForConditionalGeneration
        self.model = model_cls.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.dtype = dtype

    def __call__(self, image, max_new_tokens=30):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class Refiner:
    def __init__(self, model_name=SDXL_REFINER):
        """`model_name` is a Hugging Face repo id or a local folder with the pipeline files."""
        self.device, dtype = best_device()
        variant = "fp16" if dtype == torch.float16 else None  # the fp16 files are half the download
        try:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name, torch_dtype=dtype, variant=variant)
        except (OSError, ValueError) as err:
            print(f"fp16 weights not available ({err}); loading the default weights")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self.pipe = pipe.to(self.device)

    def __call__(self, caption, image, strength=0.3, seed=0):
        """strength: how far the refiner may move away from `image` (0 = unchanged)."""
        generator = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(caption, image=image, strength=strength, generator=generator).images[0]
