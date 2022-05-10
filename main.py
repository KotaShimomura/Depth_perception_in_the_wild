from PIL import Image
import torch
from torch.autograd import Variable
from torch.optim import RMSprop
from torchvision import transforms
from torch.backends import cudnn

from models import HourGlass
from datasets import DIW
from criterion import RelativeDepthLoss
from train_utils import fit, prep_img, save_models
import pandas as pd
import os
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CFG:
    path = '/main/workspace/Depth_perception_in_the_wild/data/DIW_Annotations/'
    data_path = '/main/workspace/Depth_perception_in_the_wild/data/'
    # path = '/tmp/working/workspace/Depth_perception_in_the_wild/data/DIW_Annotations/'
    # data_path = '/tmp/working/workspace/Depth_perception_in_the_wild/data/' 
    test_index=2
    lr=1e-3
    batch_size=24
    epoch=1
    exp_name="ex2"
    output_dir = f'./output/{exp_name}/'
    savemodel=f'./output/{exp_name}/model_{exp_name}.pth'
    im_sizew=496
    im_sizeh=496
    seed = 42
    count = 10

def seed_everything(seed=CFG.seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_dir(CFG):
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

def get_train_transforms(epoch):
    return A.Compose(
        [             
            A.Resize(CFG.im_sizeh,CFG.im_sizew),
            ToTensorV2(),
        ]
  )

def main(data_path, label_path, nb_epoch=CFG.epoch, save_path=CFG.savemodel,
         start_path=None, batch_size=CFG.batch_size, lr=CFG.lr, plot_history=True):

    seed_everything()
    make_dir(CFG)

    train = DIW(data_path, label_path, transforms=get_train_transforms(0))  
    hourglass = HourGlass()
    hourglass.cuda()
    optimizer = RMSprop(hourglass.parameters(), lr)

    if start_path:
        experiment = torch.load(start_path)
        hourglass.load_state_dict(experiment['model_state'])
        optimizer.load_state_dict(experiment['optimizer_state'])
        
    criterion = RelativeDepthLoss()
    history = fit(hourglass, train, criterion, optimizer, batch_size, nb_epoch)
    save_models(hourglass,save_path)

main(CFG.data_path+"DIW_train_val", CFG.path+"labels_train.pkl")