import os.path as path
import numpy as np
from torch.utils.data import Dataset
import albumentations as albu

def get_augmentation():
    transforms = [
        albu.RandomCrop(height=512, width=512, p=1.0)
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
            additional_targets={"comp": "image", "real": "image"})


def get_preprocessing():
    transforms = [
        albu.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        albu.Normalize(),
        albu.ToTensorV2(),
    ]
    return albu.Compose(transforms,
            additional_targets={"comp": "image", "real": "image"})


class IHarmDataset(Dataset):

    def __init__(self, root_dir, preprocessing=None, augmentation=None):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.augmentation = augmentation

        self.real_dir_path = path.join(self.root_dir, "real_images")
        self.comp_dir_path = path.join(self.root_dir, "composite_images")
        self.hist_dir_path = path.join(self.root_dir, "harmonized_results")
        self.mask_dir_path = path.join(self.root_dir, "masks")

        self.ids = os.listdir(self.comp_dir_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        comp_fname = self.ids[i]
        comp = cv2.imread(path.join(self.comp_dir_path, comp_fname))
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)

        hist = cv2.imread(path.join(self.hist_dir_path, comp_fname))
        hist = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)

        real_fname = comp_path.split("_")[0] + ".jpg"
        real = cv2.imread(path.join(self.real_dir_path, real_fname))
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

        mask_fname = "_".join(comp_path.split("_")[:-1]) + ".png"
        mask = cv2.imread(path.join(self.mask_dir_path, mask_fname))
        mask = mask[:, :, 0][:, :, np.newaxis]

        if self.augmentation:
            s = self.augmentation(comp=comp, real=real, mask=mask)
            comp, real, mask = s["comp"], s["real"], s["mask"]

        if self.preprocess:
            s = self.preprocess(comp=comp, real=real, mask=mask)
            comp, real, mask = s["comp"], s["real"], s["mask"]

        return { "comp": comp, "real": real, "mask": mask }
