import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torchsummaryX import summary

from sklearn.metrics import accuracy_score
import gc
import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime
import csv

# imports for decoding and distance calculation
import ctcdecode
from ctcdecode import CTCBeamDecoder
import Levenshtein

import warnings
warnings.filterwarnings('ignore')


from model import basic
from data_loader import load_dataset
from config import *
from data.phonemes import PHONEME_MAP, PHONEMES



def calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP):

    # h - ouput from the model. Probability distributions at each time step 
    # y - target output sequence - sequence of Long tensors
    # lh, ly - Lengths of output and target
    # decoder - decoder object which was initialized in the previous cell
    # PHONEME_MAP - maps output to a character to find the Levenshtein distance

    # TODO: You may need to transpose or permute h based on how you passed it to the criterion
    # Print out the shapes often to debug

    # TODO: call the decoder's decode method and get beam_results and out_len (Read the docs about the decode method's outputs)
    # Input to the decode method will be h and its lengths lh 
    # You need to pass lh for the 'seq_lens' parameter. This is not explicitly mentioned in the git repo of ctcdecode.
    beam_result, beam_scores, timesteps, out_len = decoder.decode(h, seq_lens=lh)

    batch_size = y.shape[0]# TODO

    dist = 0

    for i in range(batch_size): # Loop through each element in the batch

        h_sliced = beam_result[i,:out_len] # TODO: Get the output as a sequence of numbers from beam_results
        # Remember that h is padded to the max sequence length and lh contains lengths of individual sequences
        # Same goes for beam_results and out_lens
        # You do not require the padded portion of beam_results - you need to slice it with out_lens 
        # If it is confusing, print out the shapes of all the variables and try to understand

        h_string = [PHONEME_MAP.index(hh) for hh in h_sliced] # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string

        y_sliced = y[i,:ly] # TODO: Do the same for y - slice off the padding with ly
        y_string = [PHONEME_MAP.index(yy) for yy in y_sliced] # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        
        dist += Levenshtein.distance(h_string, y_string)

    dist/=batch_size

    return dist

def train(epoch, model, train_loader, optimizer, criterion):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    total_loss = 0
    total_dist = 0
    total = 0

    log_interval = 40

    for i, (x, y, lx, ly) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        total += len(y)

        x = x.to(device)
        outputs, length = model(x, lx)

        loss = criterion(outputs, y, length, ly)
        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            )

        batch_bar.update() # Update tqdm bar
        batch_bar.close() # You need this to close the tqdm bar

        train_loss = float(total_loss / total)
        lr_rate = float(optimizer.param_groups[0]['lr'])

        
    print("Epoch {}/{}: Train Loss {:.04f}, Learning Rate {:.04f}".format(
                                        epoch + 1, ARGS.epochs, train_loss, lr_rate))
        
    return train_loss, lr_rate

def validate(model, val_loader, criterion):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    total_dist = 0
    total = 0

    log_interval = 40

    for i, (x, y, lx, ly) in enumerate(tqdm(val_loader)):
        optimizer.zero_grad()
        total += len(y)

        x = x.to(device)
        outputs, length = model(x, lx)

        loss = criterion(outputs, y, length, ly)
        total_loss += loss.item()

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            )

        batch_bar.update() # Update tqdm bar
        batch_bar.close() # You need this to close the tqdm bar

        val_loss = float(total_loss / total)
        lr_rate = float(optimizer.param_groups[0]['lr'])

        
    print("Validation: Train Loss {:.04f}, Learning Rate {:.04f}".format(val_loss, lr_rate))

    return val_loss

def decode(output, seq_sizes, beam_width=60):
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, blank_id=0, beam_width=beam_width)
    output = torch.transpose(output, 0, 1) 
    probs = F.softmax(output, dim=2).data #.cpu()

    #output, scores, timesteps, out_seq_len = decoder.decode(probs=probs, seq_lens=torch.IntTensor(seq_sizes))
    output, scores, timesteps, out_seq_len = decoder.decode(probs=probs, seq_lens=seq_sizes)


    decoded = []
    for i in range(output.size(0)):
        chrs = ""
        if out_seq_len[i, 0] != 0:
            chrs = "".join(PHONEME_MAP[o] for o in output[i, 0, :out_seq_len[i, 0]])
        decoded.append(chrs)
    return decoded

def test(model, test_loader, logdir, name):
    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')

    log_result = logdir + '/submission_' + name +'.csv'
    with open(log_result, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['id', 'predictions'])
            writer.writeheader()
            cnt = 0
            with torch.no_grad():
                for batch, (x, lx) in enumerate(test_loader):
                        x = x.to(device)
                        output, length = model(x, lx)

                        decoded = decode(output, length, beam_width=100)
                        for s in decoded:
                            writer.writerow({"id": cnt, "predictions": s})
                            cnt += 1

    
if __name__ == '__main__':
    # set options for file to run
    logpath = ARGS.log_path
    logfile_base = f"{ARGS.name}_S{SEED}_B{BATCH_SIZE}_LR{LR}_E{EPOCHS}"
    logdir = logpath + logfile_base

    set_logpath(logpath, logfile_base)
    print('save path: ', logdir)

    model = basic.Network().to(device)

    # For this homework, we're limiting you to 35 million trainable parameters, as
    # outputted by this. This is to help constrain your search space and maintain
    # reasonable training times & expectations
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))

    train_loader, val_loader, test_loader = load_dataset(BATCH_SIZE)

    if ARGS.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=1e-4)
    elif ARGS.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)
        
    criterion = nn.CTCLoss().to(device)

    BEST_LOSS = 999
    best_model = model
    for epoch in range(EPOCHS):
        # You can add validation per-epoch here if you would like
        train_loss, lr_rate = train(epoch, model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        if BEST_LOSS >= val_loss:
            save_checkpoint(val_loss, model, optimizer, epoch, logdir)
            best_model = model
            BEST_LOSS = val_loss
            test(model, test_loader, logdir, str(epoch).zfill(3))
    
    test(best_model, test_loader, logdir, 'best')

