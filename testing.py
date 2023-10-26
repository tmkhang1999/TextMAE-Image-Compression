from torchvision import transforms
from utils.dataloader import get_image_dataset
from models.Compression.MCM import MCM

trainset = get_image_dataset('kodak', 'crop=224, hflip=False')

imgs, ori_shape, total_scores = trainset[0]
model = MCM()
model.update()

out_enc = model.compress(imgs.unsqueeze(dim=0), total_scores)
print(out_enc["string"])
print(out_enc["shape"])
print(out_enc["ids_restore"].shape)
out_dec = model.decompress(out_enc["string"], out_enc["shape"], out_enc["ids_restore"])
print(out_dec['x_hat'].shape)

