import argparse
import sys
import os
import time
import copy

from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
from torchvision import transforms, utils

from parse import parse_method, parser_add_main_args
from data_utils import QuickDrawDataset
import faulthandler
faulthandler.enable()

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(
    description='CS4701 Prac General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

dir_checkpoint = f"model_weights/{args.method}/"
dir_results = f"results/"
dir_logs = f"logs/"

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.cpu:
    device = torch.device('cpu')

### Add image transforms specific to each method ###
transform = None
if args.method == 'mlp':
    transform = transforms.Compose([transforms.Lambda(lambda x: x.unsqueeze(0)),
                                    transforms.Normalize((0.5, ), (0.5, )),
                                    transforms.Lambda(lambda x: torch.flatten(x))])
elif args.method == 'cnn':
    transform = transforms.Compose([transforms.Lambda(lambda x: x.expand((3, -1, -1))),
                                    transforms.Resize((28, 28)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

elif args.method == 'resnet':
    transform = transforms.Compose([transforms.Lambda(lambda x: x.expand((3, -1, -1))),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

### Load and preprocess data ###
dataset = QuickDrawDataset(transform=transform)

dataset.labels = dataset.labels.to(device)

split_idx_lst = dataset.get_idx_split()

train_loader = None, None

# print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###

model = parse_method(args, device, dataset)
model = model.to(device)

# using CrossEntropyLoss as the eval function
criterion = nn.CrossEntropyLoss()

print('MODEL:', model)

since = time.time()
### Training loop ###
for run in range(args.runs):
    #     split_idx = split_idx_lst[run]
    split_idx = split_idx_lst
    train_idx = split_idx['train'].to(device)

    # initialize training/valid/test datasets
    train_dataset = torch.utils.data.Subset(dataset, split_idx['train'])
    val_dataset = torch.utils.data.Subset(dataset, split_idx['valid'])
    test_dataset = torch.utils.data.Subset(dataset, split_idx['test'])

    dataset_sizes = {"train": len(split_idx['train']),
                     "val": len(split_idx['valid']),
                     "test": len(split_idx['test'])}

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=False, drop_last=True)

    if args.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_idx = None
    test_accs = []

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            elif phase == 'val':
                model.eval()   # Set model to evaluate mode
                data_loader = val_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = test_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (imgs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(imgs)

                    loss = criterion(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics

                running_loss += loss.item() * len(labels)
                _, preds = torch.max(preds, 1)
                running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase])
            epoch_acc = running_corrects / (dataset_sizes[phase])

            if epoch % args.display_step == 0:
                print('Phase: {} Loss: {:.4f} Acc.: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_idx = len(test_accs)

            if phase == 'test':
                test_accs.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val accuracy: {:4f}'.format(best_val_acc))
    print('Associated test accuracy: {:4f}'.format(test_accs[best_idx]))

    if args.save_cp:
        if not os.path.exists(dir_checkpoint):
            try:
                os.makedirs(dir_checkpoint, exist_ok=True)
                print('Created checkpoint directory')
            except OSError:
                pass
        model_name = datetime.today().strftime(
            f"/{args.method}%d_%b_%Y_%H_%M_%S")
        torch.save(best_model_wts,
                   dir_checkpoint + model_name + ".pth")
        print(f'Best model weights saved !')
        
        # writing arguments to log
        if not os.path.exists(dir_logs):
            try:
                os.makedirs(dir_logs, exist_ok=True)
                print('Created log directory')
            except OSError:
                pass
        
        logsname = f"logs/{model_name}"
        with open(f"{logsname}", 'a+') as write_obj:
            write_obj.write(f"{args}\n")
        
        # writing results
        if not os.path.exists(dir_results):
            try:
                os.makedirs(dir_results, exist_ok=True)
                print('Created results directory')
            except OSError:
                pass

        filename = f'results/{len(dataset.encoding)}_classes.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(f"{args.method}," +
                            f"{model_name}," +
                            "{:.0f}m {:.0f}s,".format(time_elapsed // 60, time_elapsed % 60) +
                            f"{best_val_acc}," +
                            f"{test_accs[best_idx]}\n")
