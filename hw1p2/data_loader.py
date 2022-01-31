import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn

class LibriSamples(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="dev-clean", csvpath=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 
        
        self.X_dir = data_path + "/" + partition + "/mfcc/"
        self.Y_dir = data_path + "/" + partition +"/transcript/"
        
        self.X_names = os.listdir(self.X_dir)
        self.Y_names = os.listdir(self.Y_dir)

        # using a small part of the dataset to debug
        if csvpath:
            subset = self.parse_csv(csvpath)
            self.X_names = [i for i in self.X_names if i in subset]
            self.Y_names = [i for i in self.Y_names if i in subset]
        
        if shuffle == True:
            XY_names = list(zip(self.X_names, self.Y_names))
            random.shuffle(XY_names)
            self.X_names, self.Y_names = zip(*XY_names)
        
        assert(len(self.X_names) == len(self.Y_names))
        self.length = len(self.X_names)
        
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[1])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
        
        X, Y = [], []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]
            Y_path = self.Y_dir + self.Y_names[j]
            
            label = [self.PHONEMES.index(yy) for yy in np.load(Y_path)][1:-1]

            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
            Y.append(np.array(label))
            
        X, Y = np.concatenate(X), np.concatenate(Y)
        return X, Y
    
class LibriItems(torch.utils.data.Dataset):
    def __init__(self, X, Y, context = 0):
        assert(X.shape[0] == Y.shape[0])
        
        self.length  = X.shape[0]
        self.context = context
        
        if context == 0:
            self.X, self.Y = X, Y
        else:
            #self.X, self.Y = 
            # TODO: self.X, self.Y = ... 
            pass 
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
            yy = self.Y[i]
        else:
            #xx, yy = 
            # TODO xx, yy = ...
            pass
        return xx, yy
    
class LibriTestSamples(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="test-clean", csvpath=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 
        
        self.X_dir = data_path + "/" + partition + "/mfcc/"
        self.X_order_path = "../../../data/test_order.csv"
        self.X_order = []
        
        with open(self.X_order_path) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                self.X_order.append(row[0])
            self.X_order = self.X_order[1:]

        self.length = len(self.X_order)
        
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[1])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
        
        X = []
        for j in sample_range:
            X_path = self.X_dir + self.X_order[j]

            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
            
        X = np.concatenate(X)
        return X
    
class LibriTestItems(torch.utils.data.Dataset):
    def __init__(self, X, context = 0):
        
        self.length  = X.shape[0]
        self.context = context
        
        if context == 0:
            self.X = X
        else:
            #self.X, self.Y = 
            # TODO: self.X, self.Y = ... 
            pass 
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
        else:
            #xx, yy = 
            # TODO xx, yy = ...
            pass
        return xx

def save_checkpoint(state, filename='checkpoint.pth', path='./'):
    print("Saving Model...")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    save_path = path + '/' + filename
    torch.save(state, save_path)