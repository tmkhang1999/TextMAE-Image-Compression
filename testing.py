from torchvision import transforms
from utils.dataloader import get_image_dataset
from models.Compression.MCM import MCM

cfg = {"crop": 224, "hflip": False}
trainset = get_image_dataset('kodak', cfg)

imgs, ori_shape, total_scores = trainset[0]
model = MCM()
model.update()

out_enc = model.compress(imgs.unsqueeze(dim=0), total_scores)
print(out_enc["string"])  # 12x(1,32,3,3)
print(out_enc["shape"])  # 3x3
print(out_enc["ids_restore"].shape)
out_dec = model.decompress(
    out_enc["string"], out_enc["shape"], out_enc["ids_restore"])
print(out_dec['x_hat'].shape)

