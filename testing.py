from torchvision import transforms
from utils.dataloader import get_image_dataset
from models.Compression.MCM import MCM
from torch.utils.data import DataLoader, SequentialSampler
from models.Compression.common import logger

if __name__ == '__main__':
    cfg = {"crop": 224, "hflip": False}
    test_dataset = get_image_dataset("./datasets/kodak", cfg)
    sampler_val = SequentialSampler(test_dataset)

    test_dataloader = DataLoader(
        test_dataset,
        sampler=None,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False)

    model = MCM()
    model.update()
    for (imgs, ori_shapes, total_scores) in test_dataloader:
        out = model(imgs, total_scores)

        out_enc = model.compress(imgs, total_scores)
        print(len(out_enc["string"][0]), len(out_enc["string"][1]))
        print(out_enc["shape"])  
        print(out_enc["ids_restore"].shape)
        out_dec = model.decompress(
            out_enc["string"], out_enc["shape"], out_enc["ids_restore"])
        print(out_dec["x_hat"].shape)

