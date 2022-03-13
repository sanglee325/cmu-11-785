import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np

from submit.classification import ClassificationTestSet
from config import *

"""
Transforms (data augmentation) is quite important for this task.
Go explore https://pytorch.org/vision/stable/transforms.html for more details
"""
DATA_DIR = "data"

if ARGS.half == True:
    TRAIN_DIR = osp.join(DATA_DIR, "train_subset/train_subset")
else:
    TRAIN_DIR = osp.join(DATA_DIR, "classification/classification/train")

VAL_DIR = osp.join(DATA_DIR, "classification/classification/dev")
TEST_DIR = osp.join(DATA_DIR, "classification/classification/test")

def load_dataset(batch_size):
    if ARGS.aug_type is None:
        train_transforms = [transforms.ToTensor()]
    val_transforms = [transforms.ToTensor()]

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                    transform=transforms.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                                transform=transforms.Compose(val_transforms))


    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=ARGS.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=ARGS.num_workers)

    test_dataset = ClassificationTestSet(TEST_DIR, transforms.Compose(val_transforms))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=ARGS.num_workers)

    return train_loader, val_loader, test_loader