import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp

from tqdm import tqdm
import numpy as np

from libri_dataset import LibriSamples, LibriSamplesTest
from config import *


DATA_DIR = "data"

def load_dataset(batch_size):
    train_data = LibriSamples(DATA_DIR, 'train')
    val_data = LibriSamples(DATA_DIR, 'dev')
    test_data = LibriSamplesTest(DATA_DIR, 'test_order.csv')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                                num_workers=ARGS.num_workers, collate_fn=LibriSamples.collate_fn) # TODO: Define the train loader. Remember to pass in a parameter (function) for the collate_fn argument 
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                                num_workers=ARGS.num_workers, collate_fn=LibriSamples.collate_fn) # TODO: Define the val loader. Remember to pass in a parameter (function) for the collate_fn argument 
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                                num_workers=ARGS.num_workers, collate_fn=LibriSamplesTest.collate_fn) # TODO: Define the test loader. Remember to pass in a parameter (function) for the collate_fn argument 

    print("Batch size: ", batch_size)
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    return train_loader, val_loader, test_loader