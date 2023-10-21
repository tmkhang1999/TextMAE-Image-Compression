import torch
import models_mae_new
import os


class MAE:
    def __init__(self, gan_loss=False, arch='mae_vit_large_patch16'):
        self.chkpt_dir = self.get_model_arch(gan_loss)
        self.arch = arch

    def prepare_model(self):
        # Check if the checkpoint file exists, if not, download it
        if not os.path.exists(self.chkpt_dir):
            download_link = self.get_download_link(self.chkpt_dir)
            os.system(f"wget -nc -P weights/mae/ {download_link}")

        # Build model
        model = getattr(models_mae_new, self.arch)()

        # Load model
        checkpoint = torch.load(self.chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)

        return model, msg

    @staticmethod
    def get_model_arch(gan_loss):
        if gan_loss:
            return "weights/mae/mae_vit_large_patch16_ganloss.pth"
        else:
            return "weights/mae/mae_vit_large_patch16.pth"

    @staticmethod
    def get_download_link(chkpt_dir):
        if "ganloss" in chkpt_dir:
            return "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth"
        else:
            return "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth"
