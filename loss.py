import torch
import torch.nn as nn
import numpy as np

class PoissonLoss():
    def __init__(self, device):
        w = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
        w_tensor = torch.from_numpy(w).float().view(1, 1, 3, 3).to(device)

        self.L = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.L.weight = nn.Parameter(w_tensor)
        for param in self.L.parameters():
            param.requires_grad = False

    def forward(self, pred, y):
        loss = 0
        for i in range(3):
            Lp = self.L(pred[:,i,:,:].unsqueeze(1))
            Ly = self.L(y[:,i,:,:].unsqueeze(1))
            loss += ((Lp - Ly)**2).mean()
        return loss / 3


# Not used: LPIPS is easier...
class StyleLoss():
    def __init__(self, device):
        vgg = models.vgg16(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
        vgg.to(device)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])


    def features(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]


    def compute(self, pred, gt):
        pred = F.interpolate(pred, (224, 224))
        style_pred = [gram_matrix(y) for y in self.features(pred)]

        gt = F.interpolate(gt, (224, 224))
        style_gt = [gram_matrix(y) for y in self.features(gt)]

        style_loss = 0
        for gm_pred, gm_gt in zip(style_pred, style_gt):
            style_loss += ((gm_pred - gm_gt)**2).mean()

        return style_loss
