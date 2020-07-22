import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def post(im, resize=None, bw=False, rotate=False, name="", LAB=True):
    im_ = np.transpose(im, (1, 2, 0))
    im_ = 0.5 * (im_ + 1)
    im_ = 255 * im_
    im_ = np.clip(im_, 0, 255)
    im_ = im_.astype(np.uint8)
    if not bw:
        color = cv2.COLOR_LAB2BGR if LAB else cv2.COLOR_RGB2BGR
        im_ = cv2.cvtColor(im_, color)
    if bw:
        im_ = cv2.merge((im_, im_, im_))
    if resize is not None:
        im_ = cv2.resize(im_, (resize, resize), cv2.INTER_LINEAR)
    if name != "":
        im_ = cv2.putText(im_, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return im_


def runway_post(im):
    im_ = np.transpose(im, (1, 2, 0))
    im_ = 0.5 * (im_ + 1)
    im_ = 255 * im_
    im_ = np.clip(im_, 0, 255)
    im_ = im_.astype(np.uint8)
    return im_


def get_augmentation():
    transforms = [
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
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(transforms,
            additional_targets={"real": "image", "hist": "image"})


def get_train_preprocessing():
    transforms = [
        albu.RandomSizedCrop((512, 1024), height=512, width=512),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    return albu.Compose(transforms,
            additional_targets={"real": "image", "hist": "image"})


def get_inference_preprocessing():
    transforms = [
        albu.Resize(height=1024, width=1024),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    return albu.Compose(transforms,
            additional_targets={"real": "image", "hist": "image"})


def get_runway_preprocessing():
    transforms = [
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    return albu.Compose(transforms, additional_targets={"hist": "image"})


class IHarmDataset(Dataset):

    def __init__(self, root_dir, preprocessing=None, augmentation=None, LAB=True):
        self.root_dir = root_dir
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.color = cv2.COLOR_BGR2LAB if LAB else cv2.COLOR_BGR2RGB

        self.real_dir_path = os.path.join(self.root_dir, "real_images")
        self.comp_dir_path = os.path.join(self.root_dir, "composite_images")
        self.hist_dir_path = os.path.join(self.root_dir, "harmonized_results")
        self.mask_dir_path = os.path.join(self.root_dir, "masks")

        self.ids = os.listdir(self.comp_dir_path)
        # self.ids = [f for f in os.listdir(self.comp_dir_path) if "a4386" in f]

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, i, color_transform=None):
        if color_transform is None:
            color_transform = self.color

        comp_fname = self.ids[i]
        comp = cv2.imread(os.path.join(self.comp_dir_path, comp_fname))
        comp = cv2.cvtColor(comp, color_transform)

        hist_fname = comp_fname.split(".")[0] + "_model_output.jpg"
        hist = cv2.imread(os.path.join(self.hist_dir_path, hist_fname))
        hist = cv2.cvtColor(hist, color_transform)

        real_fname = comp_fname.split("_")[0] + ".jpg"
        real = cv2.imread(os.path.join(self.real_dir_path, real_fname))
        real = cv2.cvtColor(real, color_transform)

        mask_fname = "_".join(comp_fname.split("_")[:-1]) + ".png"
        mask = cv2.imread(os.path.join(self.mask_dir_path, mask_fname))
        mask = mask[:, :, 0][:, :, np.newaxis] / 255.0

        if self.augmentation:
            s = self.augmentation(image=comp, real=real, mask=mask, hist=hist)
            comp, real, mask, hist = s["image"], s["real"], s["mask"], s["hist"]

        if self.preprocessing:
            while True:
                # Guarnatee that there's some mask before processing!
                s = self.preprocessing(image=comp, real=real, mask=mask, hist=hist)
                if s["mask"].sum() > 0:
                    comp, real, mask, hist = s["image"], s["real"], s["mask"], s["hist"]
                    break

        # Mask is input, not output! Need to transpose manually
        mask = np.transpose(mask, (2, 0, 1)).float()

        return { "comp": comp, "real": real, "mask": mask, "hist": hist }

    def get(self, i, color_transform):
        return self.__getitem__(i, color_transform)

