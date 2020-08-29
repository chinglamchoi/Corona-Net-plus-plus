import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import unet_nested
import unet_dp
from sklearn.metrics import adjusted_rand_score

from skimage import io
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random


class Covid(data.Dataset):
    def __init__(self, imgs, masks, transform=transforms.ToTensor(), img_transform=transforms.Normalize([95.32360146029684], [54.90184706924663])):
        self.imgs, self.masks = imgs, masks
        self.transform, self.img_transform = transform, img_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img1, mask1 = self.transform(self.imgs[index]).float(), self.transform(self.masks[index]).float()
        img1 = self.img_transform(img1)
        return (img1, mask1)

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))
    return alpha

def DiceLoss(a,b):
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1- ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

def RandLoss(a,b):
    a = (a>=0.5).float()
    a = a.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()
    c = adjusted_rand_score(a,b)
    c = (c+1)/2
    return 1-c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pre", metavar="PRE", type=str, default=None, dest="pre")
    parser.add_argument("-lr", metavar="LR", type=float, default=0.001, dest="lr")
    parser.add_argument("-eps", metavar="E", type=int, default=400, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=5e-4, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-batch", metavar="BATCH", type=int, default=16, dest="batch")
    parser.add_argument("-sup", metavar="SUP", type=bool, default=True, dest="sup")
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    parser.add_argument("-sam", metavar="SAM", type=bool, default=False, dest="sam")
    parser.add_argument("-ver", metavar="V", type=int, default=1, dest="ver")
    #{1: default 5 encoder unet.py, 4: 4 encoders unet_4.py}
    args = parser.parse_args()

    train_imgs, train_masks, test_imgs, test_masks = np.load("new_data/train_imgs.npy"), np.load("new_data/train_msks.npy"), np.load("new_data/test_imgs.npy"), np.load("new_data/test_msks.npy")
    sam_weights = np.load("new_data/sam_weights.npy")
    trainset = Covid(imgs=train_imgs, masks=train_masks)
    testset = Covid(imgs=test_imgs, masks=test_masks)
    bt = args.batch ###
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(sam_weights), 6556)
    trainloader = data.DataLoader(trainset, batch_size=bt, sampler=sampler, num_workers=16)
    testloader = data.DataLoader(testset, batch_size=30, shuffle=False, num_workers=16)
    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    #criterion1 = nn.BCEWithLogitsLoss().to(device)
    
    dev_id = [0,1,2,3]
    dev_id.remove(args.cuda)
    dev_id.insert(0, args.cuda)

    net = nn.DataParallel(unet_nested.run_cnn(), device_ids=dev_id)
    vall = False

    if args.pre is not None:
        checkpoint = torch.load("models_nested/" + args.pre + "/best.pt")
        net.load_state_dict(checkpoint["net"])
        vall = True #only for non-GAP pretrained
    net.to(device)
    best_loss = checkpoint["loss"] if vall else 100

    alpha = args.lr ##
    if args.opt == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=alpha, weight_decay=args.wd)
        #optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=alpha, weight_decay=args.wd, momentum=args.m)
        #optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    #if vall:
        #optimizer.load_state_dict(checkpoint["optimizer"])
        #alpha = lr_change(alpha)
    train_loss, val_loss = [], []
    train_loss1 = []
    start_ = checkpoint["epoch"] if vall else 1 
    epochs = checkpoint["epoch"]+args.eps if vall else args.eps

    for epoch in range(start_, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        net = net.train()
        epoch_loss, epoch_loss1 = 0.0, 0.0
        cnt = 0
        for img, mask in trainloader:
            img, mask = img.to(device), mask.to(device)
            mask_pred = net(img)
            t = [torch.sigmoid(i) for i in mask_pred]
            #loss = sum([criterion1(i, mask) for i in mask_pred])/4
            #epoch_loss += loss.item()
            dc_loss = sum([DiceLoss(i, mask) for i in t])/4
            epoch_loss1 += dc_loss.item()
            optimizer.zero_grad()
            dc_loss.backward()
            optimizer.step()
            cnt += 1
        #epoch_loss /= cnt
        epoch_loss1 /= cnt
        #train_loss.append(epoch_loss)
        train_loss1.append(epoch_loss1)
        #print("Epoch" + str(epoch) + " Train BCE Loss:", epoch_loss)
        print("Epoch" + str(epoch) + " Train Dice Loss:", epoch_loss1)
        
        net = net.eval()
        tot_val = 0.0
        cnt = 0
        with torch.no_grad():
            for img, mask in testloader:
                img, mask = img.to(device), mask.to(device)
                mask_pred = net(img)
                t = [torch.sigmoid(i) for i in mask_pred]
                _ = sum([DiceLoss(i, mask).item() for i in t])
                _ = _ if len(t) == 1 else _/4
                tot_val += _
                cnt += 1
        loss_ = tot_val/cnt
        print("Epoch" + str(epoch) + " Val Loss:", loss_)
        if loss_ < best_loss:
            valid = True
            print("New best test loss!")
            best_loss = loss_
        else:
            valid = False
        val_loss.append(loss_)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":net.state_dict(),
                "loss": loss_,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            path_ = ("./models_nested/" + args.opt + "_lr" + str(args.lr))
            
            path_ = path_ + "_sam" if args.sam else path_
            path_ += "/"
            try:
                os.mkdir(path_)
            except:
                pass
            torch.save(state, str(path_ + "best.pt"))
        if epoch == epochs:
            fig = plt.figure()
            plt.plot(train_loss1, label="Train")
            plt.plot(val_loss, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("Dice Loss")
            plt.title("Train-Val Loss")
            fig.savefig(path_+ "train_val.png")
            fig = plt.figure()
            plt.plot(train_loss, label="Train")
            plt.xlabel("Epochs")
            plt.ylabel("BCE Loss")
            plt.title("Train Loss")
            fig.savefig(path_ + "train.png")
            print("Saved plots")
