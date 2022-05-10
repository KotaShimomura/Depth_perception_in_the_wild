import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A


class DIW(Dataset):
    def __init__(self, path_img, path_target, transforms=None):
        self.path_img = path_img
        with open(path_target, 'rb') as f:
            self.targets = pickle.load(f)
        self.imgs = [target['name'] for target in self.targets]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        target = self.targets[index]
        img = Image.open(os.path.join(self.path_img, target['name']))
        img = img.convert('RGB')
        img = np.array(img)
        
        if self.transforms is not None:
            img = self.transforms(image = img)["image"]
        img=img.float()
        return img, target

def get_train_transforms(epoch):
    return A.Compose(
        [             
            A.Resize(CFG.im_sizeh,CFG.im_sizew),
            ToTensorV2(),
        ]
  )
