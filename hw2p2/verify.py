import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp
import argparse

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np

from model import sn
from submit.verification import VerificationDataset

from model import sn, resnet, mobilenet, mobilenetv3
from model.inception_resnet_v1 import InceptionResnetV1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0

DATA_DIR = "data"
VERI_VAL_DIR = osp.join(DATA_DIR, "verification/verification/dev")
VERI_TEST_DIR = osp.join(DATA_DIR, "verification/verification/test")

VAL_CSV = osp.join(DATA_DIR, "verification/verification/verification_dev.csv")
TEST_CSV = osp.join(DATA_DIR, "verification/verification/verification_test.csv")

def parse_args():
    # model seting options
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--model', default='sn', type=str,
                        help='model type (default: simple network)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    
    # save path
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--model_dir', default="./logs/0/", type=str,
                        help='path for results')
    parser.add_argument('--model_name', default="ckpt.pth", type=str,
                        help='path for results')
    
    parser.add_argument('--num_workers', default=0, type=int, help='num workers')

    return parser.parse_args()


def load_veri_dataset(args, batch_size):
    val_transforms = [transforms.ToTensor()]

    val_dataset = VerificationDataset(VERI_VAL_DIR, transforms.Compose(val_transforms))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             shuffle=False, num_workers=args.num_workers)

    test_dataset = VerificationDataset(VERI_TEST_DIR, transforms.Compose(val_transforms))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                              shuffle=False, num_workers=args.num_workers)

    return val_loader, test_loader


def get_feature(model, data_loader):
    model.eval()

    feats_dict = dict()
    for batch_idx, (imgs, path_names) in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=False):
        imgs = imgs.to(device)
        #print(path_names)

        with torch.no_grad():
            # Note that we return the feats here, not the final outputs
            # Feel free to try the final outputs too!
            feats = model(imgs, return_feats=True) 
        
        # TODO: Now we have features and the image path names. What to do with them?
        # Hint: use the feats_dict somehow.
        for idx, (path_name) in enumerate(path_names):
            feats_dict[path_name] = feats[idx]
        
    return feats_dict

def measure_similarity(features, csv_file, validation=True):
    similarity_metric = nn.CosineSimilarity(dim=0)
    # Now, loop through the csv and compare each pair, getting the similarity between them
    pred_similarities = []
    gt_similarities = []
    for line in tqdm(open(csv_file).read().splitlines()[1:], position=0, leave=False): # skip header
        if validation:
            img_path1, img_path2, gt = line.split(",")
            gt_similarities.append(int(gt))
        else:
            img_path1, img_path2 = line.split(",")

        # TODO: Use the similarity metric
        # How to use these img_paths? What to do with the features?
        # similarity = similarity_metric(...)
        img_name1, img_name2 = img_path1.split("/")[1], img_path2.split("/")[1]
        pred = similarity_metric(features[img_name1], features[img_name2])
        pred_similarities.append(float(pred.item()))
    

    pred_similarities = np.array(pred_similarities)
    gt_similarities = np.array(gt_similarities)

    if validation:
        print("AUC:", roc_auc_score(gt_similarities, pred_similarities))

    return pred_similarities


if __name__ == '__main__':
    # set options for file to run
    args = parse_args()

    # load model 
    NUM_CLASSES = 7000
    if args.model == 'sn':
        model = sn.Network(num_classes=NUM_CLASSES)
    elif args.model == 'resnet32':
        model = resnet.resnet32(num_classes=NUM_CLASSES)
    elif args.model == 'mobilenet':
        model = mobilenet.MobileNetV2(num_classes=NUM_CLASSES)
    elif args.model == 'mobilenetv3':
        model = mobilenetv3.mobilenetv3_large(num_classes=NUM_CLASSES)
    elif args.model == 'inceptionv1':
        model = InceptionResnetV1(classify=True, num_classes=NUM_CLASSES)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model_path = osp.join(args.model_dir, args.model_name)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['net'])
    optimizer.load_state_dict(ckpt['optimizer'])

    val_loader, test_loader = load_veri_dataset(args, args.batch_size)
    val_feats_dict = get_feature(model, val_loader)
    test_feats_dict = get_feature(model, test_loader)

    val_pred_similarities = measure_similarity(val_feats_dict, VAL_CSV)
    test_pred_similarities = measure_similarity(test_feats_dict, TEST_CSV, False)

    verification_path = osp.join(args.model_dir, "verification.csv")
    with open(verification_path, "w+") as f:
        f.write("id,match\n")
        for i in range(len(test_pred_similarities)):
            f.write("{},{}\n".format(i, test_pred_similarities[i]))


