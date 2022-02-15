import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from data_loader import LibriSamples, LibriItems
from data_loader import save_checkpoint, LibriTestSamples, LibriTestItems
from network import Network1024, Network2048

def train(args, model, device, train_samples, optimizer, criterion, epoch):
    model.train()
    for i in range(len(train_samples)):
        X, Y = train_samples[i]
        train_items = LibriItems(X, Y, context=args['context'])
        train_loader = torch.utils.data.DataLoader(train_items, batch_size=args['batch_size'], shuffle=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device)
            target = target.long().to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, dev_samples):
    model.eval()
    true_y_list = []
    pred_y_list = []
    with torch.no_grad():
        for i in range(len(dev_samples)):
            X, Y = dev_samples[i]

            test_items = LibriItems(X, Y, context=args['context'])
            test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False)

            for data, true_y in test_loader:
                data = data.float().to(device)
                true_y = true_y.long().to(device)                
                
                output = model(data)
                pred_y = torch.argmax(output, axis=1)

                pred_y_list.extend(pred_y.tolist())
                true_y_list.extend(true_y.tolist())

    train_accuracy =  accuracy_score(true_y_list, pred_y_list)
    return train_accuracy


def generate_submission(args, model, device, test_samples):
    model.eval()
    
    pred_y_list = []
    with torch.no_grad():
        for i in range(len(test_samples)):
            X = test_samples[i]

            test_items = LibriTestItems(X, context=args['context'])
            test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False)

            for data in test_loader:
                data = data.float().to(device)           
                
                output = model(data)
                pred_y = torch.argmax(output, axis=1)

                pred_y_list.extend(pred_y.tolist())

    f = open('result_2048_8_lr005_dpa01_ctx25_e5.csv','w', newline='')
    wr = csv.writer(f)
    wr.writerow(['id', 'label'])
 
    for idx, yy in enumerate(pred_y_list):
        wr.writerow([idx, yy])

    f.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_size = 3213
    model = Network2048(input_size=input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    # If you want to use full Dataset, please pass None to csvpath
    train_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="train-clean-100", csvpath=None)
    dev_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="dev-clean")

    for epoch in range(1, args['epoch'] + 1):
        train(args, model, device, train_samples, optimizer, criterion, epoch)
        test_acc = test(args, model, device, dev_samples)
        print('Dev accuracy ', test_acc)
        save_checkpoint(model,filename='checkpoint_2048_8_lr005_dpa01_ctx25_e5.pth')

    test_samples = LibriTestSamples(data_path = args['LIBRI_PATH'], shuffle=False, partition="test-clean")
    generate_submission(args, model, device, test_samples)
    
    
if __name__ == '__main__':
    args = {
        'batch_size': 2048,
        'context': 25,
        'log_interval': 1000,
        'LIBRI_PATH': '../../../data',
        'lr': 5e-4,
        'epoch': 5
    }
    main(args)
