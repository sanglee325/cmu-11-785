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
from submit.verification import VerificationDataset
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

VERI_VAL_DIR = osp.join(DATA_DIR, "verification/verification/dev")

def load_dataset(batch_size):
    train_transforms = [transforms.ToTensor()]
    val_transforms = [transforms.ToTensor()]

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                    transform=transforms.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                                transform=transforms.Compose(val_transforms))


    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=8)

    test_dataset = ClassificationTestSet(TEST_DIR, transforms.Compose(val_transforms))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=1)

    return train_loader, val_loader, test_loader


def load_veri_dataset(batch_size):
    val_transforms = [transforms.ToTensor()]

    val_veri_dataset = VerificationDataset(osp.join(DATA_DIR, "verification/verification/dev"),
                                       transforms.Compose(val_transforms))
    val_ver_loader = torch.utils.data.DataLoader(val_veri_dataset, batch_size=batch_size, 
                                             shuffle=False, num_workers=8)