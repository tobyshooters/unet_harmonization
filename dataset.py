import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

def get_augmentation():
    transforms = [
        albu.RandomCrop(height=512, width=512, p=1.0),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(transforms,
            additional_targets={"real": "image", "hist": "image"})


def get_preprocessing():
    transforms = [
        albu.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        albu.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ]
    return albu.Compose(transforms,
            additional_targets={"real": "image", "hist": "image"})


class IHarmDataset(Dataset):

    def __init__(self, root_dir, preprocessing=None, augmentation=None):
        self.root_dir = root_dir
        self.preprocessing = preprocessing
        self.augmentation = augmentation

        self.real_dir_path = os.path.join(self.root_dir, "real_images")
        self.comp_dir_path = os.path.join(self.root_dir, "composite_images")
        self.hist_dir_path = os.path.join(self.root_dir, "harmonized_results")
        self.mask_dir_path = os.path.join(self.root_dir, "masks")

        self.ids = os.listdir(self.comp_dir_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        comp_fname = self.ids[i]
        comp = cv2.imread(os.path.join(self.comp_dir_path, comp_fname))
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)

        hist_fname = comp_fname.split(".")[0] + "_model_output.jpg"
        hist = cv2.imread(os.path.join(self.hist_dir_path, hist_fname))
        hist = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)

        real_fname = comp_fname.split("_")[0] + ".jpg"
        real = cv2.imread(os.path.join(self.real_dir_path, real_fname))
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

        mask_fname = "_".join(comp_fname.split("_")[:-1]) + ".png"
        mask = cv2.imread(os.path.join(self.mask_dir_path, mask_fname))
        mask = mask[:, :, 0][:, :, np.newaxis]

        if self.augmentation:
            s = self.augmentation(image=comp, real=real, mask=mask, hist=hist)
            comp, real, mask, hist = s["image"], s["real"], s["mask"], s["hist"]

        if self.preprocessing:
            s = self.preprocessing(image=comp, real=real, mask=mask, hist=hist)
            comp, real, mask, hist = s["image"], s["real"], s["mask"], s["hist"]

        # Mask is input, not output!
        mask = np.transpose(mask, (2, 0, 1)).float()

        return { "comp": comp, "real": real, "mask": mask, "hist": hist }
