from collections import defaultdict
from shutil import copyfile

import torch
#from tqdm import tqdm_notebook
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import time
import math

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

# +
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


# -

def _fit_epoch(model, loader, criterion, optimizer, epoch, LOGGER, CFG):
    model.train()
    loss_meter = AverageMeter()
    start = end = time.time()
    t = tqdm(loader, total=len(loader))
    c = 0
    for step, (data, target) in tqdm(enumerate(loader)):
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
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(epoch+1, step, len(loader), 
                          remain=timeSince(start, float(step+1)/len(loader)),
                          loss=loss_meter))               
        if c == CFG.count:
            break
                  
    return loss_meter.avg


def fit(model, train, criterion, optimizer, LOGGER, CFG, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, cuda=True, num_workers=0):
    # TODO: implement CUDA flags, optional metrics and lr scheduler
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
    else:
        print('Train on {} samples'.format(len(train)))

    train_loader = DataLoader(train, batch_size, shuffle, num_workers=num_workers, pin_memory=True)
    t = tqdm(range(nb_epoch), total=nb_epoch)
    
    avg_loss_best = 10.
    
    for epoch in t:
        LOGGER.info(f"========== epoch: {epoch} training ==========")
        
        start_time = time.time()
        
        #train
        avg_loss = _fit_epoch(model, train_loader, criterion, optimizer, epoch, LOGGER, CFG)
        
        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        print(avg_loss)
        
        if avg_loss_best >  avg_loss:
            avg_loss_best = avg_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best loss: {avg_loss_best:.4f} Model')
            save_models(model,"./output/" + CFG.exp_name + "/model_" + CFG.exp_name + f"_epoch{epoch}.pth")
            
        LOGGER.info(f"========== epoch: {epoch} result ==========")
        LOGGER.info(f'loss: {avg_loss_best:<.4f}')
            
            
        
        #torch.cuda.empty_cache()
        #gc.collect()
        #eval
        #avg_val_loss = _fit_epoch(model, train_loader, criterion)

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
