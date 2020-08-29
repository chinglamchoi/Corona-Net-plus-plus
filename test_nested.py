import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
#import unet_4
import unet_nested
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

def DiceLoss(a,b):
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

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
    parser.add_argument("-eps", metavar="E", type=int, default=400, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=5e-4, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-mul", metavar="MUL", type=bool, default=False, dest="mul") #same thing as GAP
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    parser.add_argument("-sam", metavar="SAM", type=bool, default=False, dest="sam")
    parser.add_argument("-ver", metavar="V", type=int, default=6, dest="ver")
    #{1: default 5 encoder unet.py, 4: 4 encoders unet_4.py}
    args = parser.parse_args()

    test_imgs, test_masks = np.load("data/mosmed_test_imgs.npy"), np.load("data/mosmed_test_masks.npy")
    testset = Covid(imgs=test_imgs, masks=test_masks)

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=12)

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(unet_nested.run_cnn(), device_ids=[0,1,2,3])
    checkpoint = torch.load("models_nested/" + args.pre + "/best.pt")
    net.load_state_dict(checkpoint["net"])
    net.to(device)
    net = net.eval()
    tot_val = 0.0
    tot_rand = 0.0
    countt = 0
    try:
        os.mkdir("mask_pred/" + args.pre)
    except:
        pass
    with torch.no_grad():
        for img, mask in testloader:
            mask_type = torch.float32
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            mask_pred = net(img)
            t = [torch.sigmoid(i) for i in mask_pred]
            tot_val += sum([DiceLoss(i, mask).item() for i in t])
            tot_rand += sum([RandLoss(i, mask) for i in t])
            cnt1 = 0
            for i in mask_pred:
                i.to("cpu")
                torchvision.utils.save_image(i, "mask_pred/" + args.pre + "/" + str(countt) + "_" + str(cnt1) + ".jpg")
                cnt1 += 1
            countt += 1
    print("Dice Accuracy:", tot_val/countt)
    print("Rand Loss:", tot_rand/countt)
