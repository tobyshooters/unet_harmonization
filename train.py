# TODO:
# Use basic U-net from milesial
# Train on fixed-size patches, randomly sampled/scaled/cropped from image
# Loss: L1, try L2 or SSIM or perceptual or GAN

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from unet import UNet
from dataset import IHarmDataset, get_preprocessing, get_augmentation

def train(net, device, epochs, batch_size, lr, num_workers, save_cp):

    preprocessing = get_preprocessing()
    augmentation = get_augmentation()
    dataset = IHarmDataset("./HAdobe5k", preprocessing, augmentation)

    n_train = int(0.8 * len(dataset))
    train, val = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # criterion = nn.L1Loss() 

    # for e, epoch in enumerate(epochs):
    #     net.train()

if __name__ == '__main__':

    net = UNet(n_channels=7, n_classes=3)
    train(net, "cpu", 1, 4, 0.1, 0, False)
