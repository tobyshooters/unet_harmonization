# TODO:
# 1. Need to load in histograms, not currently doing
# 2. Problem: most crops will not have the mask within them
# 3. Loss: L1, try perceptual or ssim
# 4. Validation metric
# 5. How to use scheduler right
# 6. Use tensorboard to get preview images

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from dataset import IHarmDataset, get_preprocessing, get_augmentation
from utils import MovingAverage

def evaluate(net, loader, device):
    net.eval()
    total_loss = 0
    n_val = len(loader)

    for batch in loader:
        x = torch.cat([batch["comp"], batch["mask"], batch["hist"]], dim=1)
        x = x.to(device)

        with torch.no_grad():
            pred = net(x)

        y = batch["real"]
        y = y.to(device)

    net.train()
    return 


def train(net, device, epochs, batch_size, lr, num_workers, save_cp):

    dataroot = "../image_harmonization/HAdobe5k/"
    preprocessing = get_preprocessing()
    augmentation = get_augmentation()
    dataset = IHarmDataset(dataroot, preprocessing, augmentation)

    n_train = int(0.9 * len(dataset))
    train, val = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # criterion_style = StyleLoss(layers=("relu3_3"))
    criterion_l1 = nn.L1Loss() 
    # criterion_ssim = SSIMLoss(data_range=1.)

    writer = SummaryWriter()
    avg_loss = MovingAverage()
    n_iter = 0

    for epoch in range(epochs):
        net.train()

        pbar = tqdm(train_loader)
        for batch in pbar:
            c, m, h = batch["comp"], batch["mask"], batch["hist"]
            pred = net(c.to(device), m.to(device), h.to(device))

            y = batch["real"]
            y = y.to(device)

            loss = criterion_l1(pred, y)
            # loss_ssim = criterion_ssim(pred, y)
            # loss_style = criterion_style(pred, y)
            # losses = [loss_l1, loss_style]
            # loss = torch.sum(losses)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            _loss = loss.item()
            avg_loss.update(_loss)
            pbar.set_description(f'Loss: {_loss}')
            writer.add_scalar('L1/Train', _loss, n_iter)

            n_iter += 1

        scheduler.step()

        if save_cp:
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")
            path = "{}/cp_masked_epoch_{}.pth".format("checkpoints", epoch+1)
            torch.save(net.state_dict(), path)

    writer.close()


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs = 10
    batch_size = 8
    learning_rate = 0.1
    num_workers = 4
    save_checkpoint = True
    init_checkpoint = "checkpoints/cp_best_l1_ssim.pth"

    net = UNet(n_channels=7, n_classes=3).to(device)
    net.load_state_dict(torch.load(init_checkpoint))

    try:
        train(net, device, epochs, batch_size, learning_rate, num_workers, save_checkpoint)
    except KeyboardInterrupt:
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        path = "{}/cp_epoch_{}.pth".format("checkpoints", "keyboard")
        torch.save(net.state_dict(), path)
