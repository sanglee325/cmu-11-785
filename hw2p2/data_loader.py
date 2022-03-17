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


DATA_DIR = "data"

if ARGS.half == True:
    TRAIN_DIR = osp.join(DATA_DIR, "train_subset/train_subset")
else:
    TRAIN_DIR = osp.join(DATA_DIR, "classification/classification/train")

VAL_DIR = osp.join(DATA_DIR, "classification/classification/dev")
TEST_DIR = osp.join(DATA_DIR, "classification/classification/test")

def load_dataset(batch_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if ARGS.aug_type is None:
        train_transforms = transforms.Compose([
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=mean, std=std)
                            ])
    if ARGS.aug_type == 'basic':
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    val_transforms = transforms.Compose([
                        transforms.ToTensor(), 
                        transforms.Normalize(mean=mean, std=std)
                        ])

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                    transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                                transform=val_transforms)


    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=ARGS.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=ARGS.num_workers)

    test_dataset = ClassificationTestSet(TEST_DIR, val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=ARGS.num_workers)

    return train_loader, val_loader, test_loader