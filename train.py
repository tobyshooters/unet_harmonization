# TODO:
# Check perceptual loss in correct

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from piq import SSIMLoss
from lpips import PerceptualLoss
from poisson import PoissonLoss

from unet import UNet, HistNet
from dataset import IHarmDataset, get_train_preprocessing, get_augmentation
from utils import MovingAverage

torch.manual_seed(3)
np.random.seed(3)

def eval(net, loader, device):
    net.eval()
    total_loss = 0
    nval = len(loader)

    with tqdm(loader, desc='Val', leave=True) as pbar:
        for batch in pbar:
            c = batch["comp"].float().to(device)
            m = batch["mask"].float().to(device)
            h = batch["hist"].float().to(device)
            y = batch["real"].float().to(device)

            with torch.no_grad():
                pred = net(c, m, h)

            total_loss += F.l1_loss(pred, y)
            pbar.update()

    net.train()
    return total_loss / nval


def train(net, device, epochs, batch_size, lr, num_workers, save_cp, checkpoint_name):
    dataroot = "../image_harmonization/HAdobe5k/"
    preprocessing = get_train_preprocessing()
    augmentation = get_augmentation()
    dataset = IHarmDataset(dataroot, preprocessing, augmentation)

    n_train = int(0.9 * len(dataset))
    train, val = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

    criterion_l1 = nn.L1Loss() 
    # criterion_ssim = SSIMLoss()
    criterion_poisson = PoissonLoss(device)
    criterion_lpips = PerceptualLoss(
            model='net-lin', net='alex', use_gpu=(device != "cpu"), gpu_ids=[0])

    writer = SummaryWriter()
    avg_l1 = MovingAverage()
    avg_poisson = MovingAverage()
    avg_style = MovingAverage()
    avg_loss = MovingAverage()
    n_iter = 0

    print("val test")
    val_score = eval(net, val_loader, device)

    for epoch in range(epochs):
        net.train()

        pbar = tqdm(train_loader)
        for batch in pbar:
            c = batch["comp"].float().to(device)
            m = batch["mask"].float().to(device)
            h = batch["hist"].float().to(device)
            y = batch["real"].float().to(device)

            pred = net(c, m, h)

            loss_l1 = criterion_l1(pred, y)
            loss_poisson = 1e2 * criterion_poisson.forward(pred, y)
            loss_style = criterion_lpips.forward(pred, y).mean()
            loss = loss_l1 + loss_poisson + loss_style

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            _loss, _l1, _poisson, _style = loss.item(), loss_l1.item(), loss_poisson.item(), loss_style.item()
            avg_l1.update(_l1)
            avg_poisson.update(_poisson)
            avg_style.update(_style)
            avg_loss.update(_loss)
            pbar.set_description(f'Poisson: {avg_poisson.get()}, L1: {avg_l1.get()}, Style: {avg_style.get()}')
            writer.add_scalar('Loss/Train', _loss, n_iter)
            writer.add_scalar('Poisson/Train', _poisson, n_iter)
            writer.add_scalar('L1/Train', _l1, n_iter)
            writer.add_scalar('Style/Train', _style, n_iter)
            n_iter += 1

        # Calcuate validation
        val_score = eval(net, val_loader, device)
        scheduler.step(val_score)

        if save_cp:
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")
            path = "{}/cp_{}_epoch_{}.pth".format("checkpoints", checkpoint_name, epoch+1)
            torch.save(net.state_dict(), path)

    writer.close()


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs = 10
    batch_size = 8
    learning_rate = 3e-4
    num_workers = 4
    save_checkpoint = True
    init_checkpoint = "checkpoints/cp_HistNet_identity.pth" 
    checkpoint_name = "HistNet"

    # net = UNet(n_channels=7, n_classes=3).to(device)
    net = HistNet().to(device)

    if init_checkpoint is not None:
        net.load_state_dict(torch.load(init_checkpoint))

    print("Training! Checkpoint: {}, Device: {}".format(device, init_checkpoint))
    train(net, device, epochs, batch_size, learning_rate, num_workers, save_checkpoint, checkpoint_name)
