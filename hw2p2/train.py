import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np

from model import sn

from data_loader import load_dataset
from config import *


def train(model, train_loader, optimizer, scheduler, criterion, scaler, batch_size):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0
    total = 0

    for i, (x, y) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        # Don't be surprised - we just wrap these two lines to make it work for FP16
        with torch.cuda.amp.autocast():     
            outputs = model(x)
            loss = criterion(outputs, y)

        # Update # correct & loss as we go
        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total += len(x)
        total_loss += float(loss)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        
        # Another couple things you need for FP16. 
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

        batch_bar.update() # Update tqdm bar
        batch_bar.close() # You need this to close the tqdm bar

        train_acc = 100 * num_correct / total
        train_loss = float(total_loss / total)
        lr_rate = float(optimizer.param_groups[0]['lr'])

        
    print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
                                        epoch + 1, ARGS.epochs, train_acc, train_loss, lr_rate))
        
    return train_acc, train_loss, lr_rate

def validate(model, val_loader, batch_size):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    num_correct = 0
    total = 0
    for i, (x, y) in enumerate(val_loader):

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            outputs = model(x)

        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total += len(x)
        batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)))

        batch_bar.update()
        
    batch_bar.close()

    val_acc = 100 * num_correct / total
    print("Validation: {:.04f}%".format(100 * num_correct / total))

    return val_acc

def test(model, test_loader, logdir, name):
    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')

    res = []
    total = 0
    for i, (x) in enumerate(test_loader):
        x = x.to(device)

        with torch.no_grad():
            outputs = model(x)
        total += len(x)
        pred = torch.argmax(outputs, axis=1)
        res += pred        
        
        batch_bar.update()
        
    batch_bar.close()

    log_result = logdir + '/result_' + name +'.csv'
    with open(log_result, "w+") as f:
        f.write("id,label\n")
        for i in range(len(res)):
            f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", res[i]))

    
if __name__ == '__main__':
    # set options for file to run
    logpath = ARGS.log_path
    logfile_base = f"{ARGS.name}_{ARCH}_S{SEED}_B{BATCH_SIZE}_LR{LR}_E{EPOCHS}"
    logdir = logpath + logfile_base

    set_logpath(logpath, logfile_base)
    print('save path: ', logdir)
    # define model
    # model = torchvision.models.__dict__[ARCH](num_classes=7000)
    model = sn.Network(num_classes=7000)
    model.to(device)

    # For this homework, we're limiting you to 35 million trainable parameters, as
    # outputted by this. This is to help constrain your search space and maintain
    # reasonable training times & expectations
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))

    train_loader, val_loader, test_loader = load_dataset(BATCH_SIZE)
 
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader)*EPOCHS))
    criterion = nn.CrossEntropyLoss().to(device)


    # T_max is "how many times will i call scheduler.step() until it reaches 0 lr?"

    # For this homework, we strongly strongly recommend using FP16 to speed up training.
    # It helps more for larger models.
    # Go to https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision
    # and compare "Single precision training" section with "Mixed precision training" section
    scaler = torch.cuda.amp.GradScaler()

    BEST_VAL = 0
    best_model = model
    for epoch in range(EPOCHS):
        # You can add validation per-epoch here if you would like
        train_acc, train_loss, lr_rate = train(model, train_loader, optimizer, scheduler,
                                                     criterion, scaler, BATCH_SIZE)
        val_acc = validate(model, val_loader, BATCH_SIZE)
        if BEST_VAL <= val_acc:
            save_checkpoint(val_acc, model, optimizer, epoch, logdir)
            best_model = model
            BEST_VAL = val_acc
            test(model, test_loader, logdir, str(epoch).zfill(3))
    
    test(best_model, test_loader, logdir, 'best')

