import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
import Levenshtein as lev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn

#import seaborn as sns
#import matplotlib.pyplot as plt
import time
import random
import datetime
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm

# imports for decoding and distance calculation
import Levenshtein

from models.seq2seq import Seq2Seq
from data_loader import create_dictionaries, load_dataset
from letter_list import LETTER_LIST


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

def save_checkpoint(loss, model, optim, epoch, logdir, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_' + str(epoch).zfill(3) + '_' + str(SEED) + '.pth'
    else:
        ckpt_name = 'ckpt_' + str(SEED).zfill(4) + '.pth'

    ckpt_path = os.path.join(logdir, ckpt_name)
    torch.save(state, ckpt_path)

def parse_args():
    # model seting options
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    parser.add_argument('--model', default='basic', type=str,
                        help='model type (default: basic network)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int,
                        help='total epochs to run')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--decay', default=2e-4, type=float, help='weight decay')
    
    # save path
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--log_path', default="./logs/", type=str,
                        help='path for results')
    
    parser.add_argument('--num_workers', default=4, type=int, help='num workers')

    parser.add_argument('--loss_type', default='ctc', type=str, help='Loss Functionn')
    parser.add_argument('--optim', default='adam', type=str, help='Optimizer')

    
    parser.add_argument('--input_dim', default=20, type=int,
                        help='total epochs to run')

    return parser.parse_args()

cuda = torch.cuda.is_available()
print(cuda, sys.version)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
    num_workers = 4
else:
    N_GPUS = 0
    num_workers = 0
    
print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))
np.random.seed(11785)
torch.manual_seed(11785)

ARGS = parse_args()

BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
LR = ARGS.lr
ARCH = ARGS.model

SEED = set_reproducibility(ARGS.seed)

letter2index, index2letter = create_dictionaries(LETTER_LIST)



