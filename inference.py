from PIL import Image
import torch
import numpy as np
from utils.mae_preprocessing import process, calculate_patch_score, get_filtered_indices
from models import Mae, Diffuser, Blip2

mae_model, _ = Mae().prepare_model(gan_loss=True)
diffusion_model = Diffuser()
diffusion_model.prepare_model()
blip_model = Blip2()
blip_model.prepare_model()


def main(img_path, mask_ratio=0.8):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # Load image
    orig_img = Image.open(img_path)

    # Generate prompt
    caption = blip_model.generate_caption(orig_img)

    # Preprocess image
    img, shape = process(orig_img, imagenet_mean, imagenet_std)
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # Encode
    total_score = calculate_patch_score(img)
    ids_shuffle = get_filtered_indices(total_score, 1 - mask_ratio)
    feature_map, _, ids_restore = mae_model.forward_encoder(x.float(), mask_ratio, ids_shuffle)

    # Decode
    pred = mae_model.forward_decoder(feature_map, ids_restore)
    pred = mae_model.unpatchify(pred)
    pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

    # Postprocess pred
    img_recon = ((pred.squeeze(dim=0) * imagenet_std + imagenet_mean) * 255).numpy().astype(np.uint8)
    img_new = Image.fromarray(img_recon).resize(shape, Image.BICUBIC)

    img_final = diffusion_model.refine_image(img_new, caption)
    return img_final


