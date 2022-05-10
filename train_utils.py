from collections import defaultdict
from shutil import copyfile

import torch
#from tqdm import tqdm_notebook
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

def prep_img(img):
    return Variable(img.unsqueeze(0)).cuda()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _fit_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_meter = AverageMeter()
    t = tqdm(loader, total=len(loader))
    c = 0
    for data, target in tqdm(t):
        data = Variable(data.cuda())
        target['x_A'] = target['x_A'].cuda()
        target['y_A'] = target['y_A'].cuda()
        target['x_B'] = target['x_B'].cuda()
        target['y_B'] = target['y_B'].cuda()
        target['ordinal_relation'] = Variable(target['ordinal_relation']).cuda()
        output = model(data)
        loss = criterion(output, target)
        loss_meter.update(loss.data)
        t.set_description("[ loss: {:.4f} ]".format(loss_meter.avg))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        c += 1
        # if c ==10:
        #     break

    return loss_meter.avg


def fit(model, train, criterion, optimizer, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, cuda=True, num_workers=0):
    # TODO: implement CUDA flags, optional metrics and lr scheduler
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
    else:
        print('Train on {} samples'.format(len(train)))

    train_loader = DataLoader(train, batch_size, shuffle, num_workers=num_workers, pin_memory=True)
    t = tqdm(range(nb_epoch), total=nb_epoch)
    for epoch in t:
        _fit_epoch(model, train_loader, criterion, optimizer)

def validate(model, validation_data, criterion, batch_size):
    model.eval()
    val_loss = AverageMeter()
    loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    for data, target in loader:
        data = Variable(data.cuda())
        target['x_A'] = target['x_A'].cuda()
        target['y_A'] = target['y_A'].cuda()
        target['x_B'] = target['x_B'].cuda()
        target['y_B'] = target['y_B'].cuda()
        target['ordinal_relation'] = Variable(target['ordinal_relation']).cuda()
        output = model(data)
        loss = criterion(output, target)
#         val_loss.update(loss.data[0])
        val_loss.update(loss.data)
    return val_loss.avg

def save_models(model, filename, is_best=False):
    torch.save(model.state_dict(),filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')
