import os
import csv

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

from letter_list import LETTER_LIST

class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, partition= "train"): # You can use partition to specify train or dev

        self.X_dir = os.path.join(data_path, partition, "mfcc")
        self.Y_dir = os.path.join(data_path, partition, "transcript")
        
        self.X_files = os.listdir(self.X_dir)
        self.Y_files = os.listdir(self.Y_dir)

        self.LETTER_LIST = LETTER_LIST

        assert(len(self.X_files) == len(self.Y_files))


    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):
    
        X = np.load(os.path.join(self.X_dir, self.X_files[ind]))
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X = torch.FloatTensor(X)
       
        Y = np.load(os.path.join(self.Y_dir, self.Y_files[ind]))
        labels = np.asarray([self.LETTER_LIST.index(yy) for yy in Y[1:-1]]) 
        
        Yy = torch.LongTensor(labels)

        return X, Yy
    
    def collate_fn(batch):
        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        batch_x_pad = pad_sequence(batch_x, batch_first=True)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [sample.shape[0] for sample in batch_x] # TODO: Get original lengths of the sequence before padding

        batch_y_pad = pad_sequence(batch_y, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [sample.shape[0] for sample in batch_y] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


# You can either try to combine test data in the previous class or write a new Dataset class for test data
class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order): # test_order is the csv similar to what you used in hw1

        test_order_list = os.path.join(data_path, "test", test_order)
        self.X_dir = os.path.join(data_path, "test", "mfcc")
        self.X_files = []

        with open(test_order_list) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                self.X_files.append(row[0])
            self.X_files = self.X_files[1:]

    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, ind):
    
        X = np.load(os.path.join(self.X_dir, self.X_files[ind])) # TODO: Load the mfcc npy file at the ind in the directory
        X = (X - X.mean(axis=0)) / X.std(axis=0) # ADD: normalize
        X = torch.FloatTensor(X)
        
        return X
    
    def collate_fn(batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [x.shape[0] for x in batch_x] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x)


if __name__ == '__main__':
    root = "data/hw4p2_student_data/hw4p2_student_data"
    batch_size = 128

    train_dataset = LibriSamples(root, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, collate_fn=LibriSamples.collate_fn)
    print("Train dataset samples = {}, batches = {}".format(train_dataset.__len__(), len(train_loader)))

    for data in train_loader:
        x, y, lx, ly = data
        print(x.shape, y.shape, lx.shape, ly.shape)
        print(y[0])
        break

    test_dataset = LibriSamplesTest(root, 'test_order.csv')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, collate_fn=LibriSamplesTest.collate_fn)
    print("Test dataset samples = {}, batches = {}".format(test_dataset.__len__(), len(test_loader)))

    for data in test_loader:
        x, lx = data
        print(x.shape, lx.shape)
    