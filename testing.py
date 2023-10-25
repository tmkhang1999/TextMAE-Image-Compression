from torchvision import transforms
from utils.dataloader import get_image_dataset

trainset = get_image_dataset('kodak', 'crop=224, hlip=True')

transform = transforms.ToPILImage()
img = transform(trainset[1][0])
img.show()
# print(f'Training transform: \n{str(trainset.transform)}')