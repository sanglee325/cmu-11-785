import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os
import shutil
import argparse
import random

import numpy as np

def parse_args():
    # model seting options
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model', default='sn', type=str,
                        help='model type (default: simple network)')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int,
                        help='total epochs to run')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--decay', default=2e-4, type=float, help='weight decay')
    
    # save path
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--log_path', default="./logs/", type=str,
                        help='path for results')
    
    parser.add_argument('--half', action='store_true', help='')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers')

    
    parser.add_argument('--aug_type', default=None, type=int, help='Data augmentation for Recognition')

    return parser.parse_args()


def set_reproducibility(seed):
    if ARGS.seed is None:
        seed = np.random.randint(10000)

    if N_GPUS == 1:
        torch.cuda.manual_seed(seed)
    else:    
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    return seed


def set_logpath(dirpath, fn):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    logdir = dirpath + fn
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if len(os.listdir(logdir)) != 0:
        ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                        "Will you proceed [y/N]? ")
        if ans in ['y', 'Y']:
            shutil.rmtree(logdir)
        else:
            exit(1)

def save_checkpoint(acc, model, optim, epoch, logdir, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(SEED) + '.pth'
    else:
        ckpt_name = 'ckpt_' + str(SEED) + '.pth'

    ckpt_path = os.path.join(logdir, ckpt_name)
    torch.save(state, ckpt_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0

ARGS = parse_args()

BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
LR = ARGS.lr
ARCH = ARGS.model

SEED = set_reproducibility(ARGS.seed)
