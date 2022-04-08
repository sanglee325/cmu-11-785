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


from model import basic, biLSTM, ink
from data_loader import load_dataset
from config import *
from phonemes import PHONEME_MAP, PHONEMES



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
    h = h.permute(1,0,2)
    beam_result, beam_scores, timesteps, out_len = decoder.decode(h, seq_lens=lh)

    batch_size = beam_scores.shape[0]# TODO

    dist = 0

    for i in range(batch_size): # Loop through each element in the batch

        score_max_idx = torch.argmax(beam_scores[i])
        h_sliced = beam_result[i][score_max_idx][:out_len[i][score_max_idx]] # TODO: Get the output as a sequence of numbers from beam_results
        # Remember that h is padded to the max sequence length and lh contains lengths of individual sequences
        # Same goes for beam_results and out_lens
        # You do not require the padded portion of beam_results - you need to slice it with out_lens 
        # If it is confusing, print out the shapes of all the variables and try to understand

        h_string = [str(PHONEME_MAP[hh]) for hh in h_sliced] # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        h_string = "".join(h_string)

        y_sliced = y[i,:ly[i]] # TODO: Do the same for y - slice off the padding with ly
        y_string = [str(PHONEME_MAP[yy])for yy in y_sliced] # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        y_string = "".join(y_string)
        
        dist += Levenshtein.distance(h_string, y_string)

    dist/=batch_size

    return dist

def train(epoch, model, train_loader, optimizer, criterion):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    total_loss = 0

    for i, (x, y, lx, ly) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(device)
        outputs, length = model(x, lx)

        loss = criterion(outputs, y, length, ly)
        loss.backward()
        total_loss += float(loss)

        optimizer.step()
        optimizer.zero_grad()

        lr_rate = float(optimizer.param_groups[0]['lr'])

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(loss),
            lr="{:.04f}".format(lr_rate)
        )
        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar
    
    train_loss = total_loss / len(train_loader)    
    print("Epoch {}/{}: train loss {:.04f}, learning rate {:.04f}".format(
                                        epoch + 1, ARGS.epochs, train_loss, lr_rate))
        
    return train_loss, lr_rate

def validate(epoch, model, val_loader, criterion, decoder):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Validation')

    total_loss = 0
    total_dist = 0

    for i, (x, y, lx, ly) in enumerate(val_loader):
        x = x.to(device)
        
        with torch.no_grad():
            outputs, length = model(x, lx)
        dist = calculate_levenshtein(outputs,y,lx,ly,decoder,PHONEME_MAP=PHONEME_MAP)
        total_dist += dist

        loss = criterion(outputs, y, length, ly)
        total_loss += loss

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            loss="{:.04f}".format(loss),
            dist="{:.04f}".format(dist)
        )

        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar

    val_loss = float(total_loss / len(val_loader))
    val_dist = float(total_dist / len(val_loader))
        
    print("Epoch {}/{}: validation loss {:.04f}, distance {:.04f}".format(epoch + 1, ARGS.epochs, val_loss, val_dist))

    return val_loss, val_dist

def make_predictions(h, lh, decoder, PHONEME_MAP):
    h = h.permute(1,0,2)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(h,seq_lens=lh)
    batch_size = beam_results.shape[0]

    h_string_list = []
    for i in range(batch_size): # Loop through each element in the batch
        idx = torch.argmax(beam_scores[i])
        h_sliced = beam_results[i][idx][:out_lens[i][idx]]

        h_string = [str(PHONEME_MAP[hh]) for hh in h_sliced] # TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        h_string = "".join(h_string)
        h_string_list.append(h_string)
    
    return h_string_list

def test(model, decoder, test_loader, logdir, name):
    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')

    log_result = logdir + '/submission_' + name +'.csv'
    with open(log_result, 'w') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(["id","predictions"])
        cnt = 0
        with torch.no_grad():
            for batch, (x, lx) in enumerate(test_loader):
                x = x.to(device)
                output, length = model(x, lx)
                predictions = make_predictions(output,lx,decoder,PHONEME_MAP)
                for prediction in predictions :
                    wr.writerow([cnt,prediction])
                    cnt += 1

    
if __name__ == '__main__':
    # set options for file to run
    logpath = ARGS.log_path
    logfile_base = f"{ARGS.name}_S{SEED}_B{BATCH_SIZE}_LR{LR}_E{EPOCHS}"
    logdir = logpath + logfile_base

    set_logpath(logpath, logfile_base)
    print('save path: ', logdir)

    if ARGS.model == 'basic':
        model = basic.Network().to(device)
    elif ARGS.model == 'biLSTM':
        model = biLSTM.Network().to(device)

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
    #scaler = torch.cuda.amp.GradScaler()

    decoder = CTCBeamDecoder(labels=PHONEME_MAP,log_probs_input=True,beam_width=1)
    test_decoder = CTCBeamDecoder(labels=PHONEME_MAP,log_probs_input=True,beam_width=100)

    BEST_LOSS = 999
    BEST_DIST = np.Inf
    best_model = None
    val_interval = 5
    for epoch in range(EPOCHS):
        train_loss, lr_rate = train(epoch, model, train_loader, optimizer, criterion)
        if (epoch+1) % val_interval == 0:
            val_loss, val_dist = validate(epoch, model, val_loader, criterion, decoder)
            if BEST_DIST >= val_dist:
                save_checkpoint(val_loss, model, optimizer, epoch, logdir, index=True)
                best_model = model
                BEST_LOSS = val_loss
                BEST_DIST = val_dist
            #test(model, test_decoder, test_loader, logdir, str(epoch).zfill(3))
    _, _ = validate(epoch, model, val_loader, criterion, test_decoder)
    test(best_model, test_decoder, test_loader, logdir, 'best')

