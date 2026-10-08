"""Image captioning with BLIP or BLIP-2: describe the original image in one sentence (the "Text" in TextMAE)."""
from transformers import (AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration,
                          BlipProcessor)

from extras.device import best_device

DEFAULT_CAPTIONER = "Salesforce/blip-image-captioning-large"  # ~1.9 GB; BLIP-2 needs ~15 GB


class Captioner:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device, self.dtype = best_device()

    def prepare_model(self, model_name=DEFAULT_CAPTIONER):
        """Load BLIP (default) or a BLIP-2 checkpoint such as Salesforce/blip2-opt-2.7b."""
        if "blip2" in model_name.lower():
            self.processor = AutoProcessor.from_pretrained(model_name)
            model_cls = Blip2ForConditionalGeneration
        else:
            self.processor = BlipProcessor.from_pretrained(model_name)
            model_cls = BlipForConditionalGeneration
        self.model = model_cls.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)

    def generate_caption(self, image, max_new_tokens=30):
        if self.model is None:
            raise RuntimeError("Call prepare_model() before generate_caption()")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
