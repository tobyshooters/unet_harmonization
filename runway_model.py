import os
import cv2
import numpy as np
import runway
import torch

from dataset import get_runway_preprocessing, runway_post
from unet import MaskAttentionUNet 
from iharm.inference.predictor import Predictor
from iharm.mconfigs import ALL_MCONFIGS

unet_preprocess = get_runway_preprocessing()

def run_harmonization(model, og_image, og_mask):
    """
    Wrapper for Harmonizatoin model.
    Resizes inputs to 256 & correct type, resizes output to orignal size.
    """
    image_size = og_image.shape[:2]
    image = cv2.resize(og_image, (256, 256), cv2.INTER_LINEAR)

    mask = cv2.resize(og_mask, (256, 256), cv2.INTER_LINEAR)
    mask = (mask[:, :, 0] > 100).astype(np.float32)

    iharm_output = model.predict(image, mask)
    iharm_output = cv2.resize(iharm_output, image_size[::-1], cv2.INTERLINEAR)
    return iharm_output


def run_color_transfer(model, og_image, og_mask, og_hist):
    """
    Wrapper for UNet color transfer model.
    Does the preprocessing done in training (see train.py & dataset.py).
    """
    image = cv2.cvtColor(og_image, cv2.COLOR_RGB2LAB)
    hist = cv2.cvtColor(og_hist, cv2.COLOR_RGB2LAB)
    mask = (og_mask[:, :, 0] > 100).astype(np.float32)
    mask = mask[:, :, np.newaxis]
    
    s = unet_preprocess(image=image, mask=mask, hist=hist)
    c, m, h = s["image"], s["mask"], s["hist"]
    m = np.transpose(m, (2, 0, 1)).float()

    result = model(c.to(device), m.to(device), h.to(device))

    harmonized = runway_post(result["output"].detach().cpu().numpy()[0])
    return harmonized


options = {
    "iharm_checkpoint": runway.file(extension=".pth"),
    "unet_checkpoint": runway.file(extension=".pth"),
}

@runway.setup(options=options)
def setup(opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup iharm network
    iharm_config = ALL_MCONFIGS["hrnet18_idih256"]
    iharm_cp = torch.load(opts['iharm_checkpoint'], map_location='cpu')
    iharm_net = iharm_config['model'](**iharm_config['params'])
    iharm_state = iharm_net.state_dict()
    iharm_state.update(iharm_cp)
    iharm_net.load_state_dict(iharm_state)
    iharm_model = Predictor(iharm_net, device)

    # Setup color transfer network
    unet_model = MaskAttentionUNet().to(device)
    unet_model_cp = torch.load(opts['unet_checkpoint'], map_location='cpu')
    unet_model.load_state_dict(unet_model_cp)

    return (iharm_model, unet_model)


inputs = {
    'composite_image': runway.image,
    'foreground_mask': runway.image,
}

outputs = {
    'harmonized_image': runway.image,
}

@runway.command('harmonize', inputs=inputs, outputs=outputs)
def harmonize(models, inputs):
    iharm_model, unet_model = models
    og_image = np.array(inputs["composite_image"])
    og_mask = np.array(inputs["foreground_mask"])

    # Chain the harmonization and color transfer models
    colors = run_harmonization(iharm_model, og_image, og_mask)
    harmonized = run_color_transfer(unet_model, og_image, og_mask, colors)

    return harmonized


if __name__ == '__main__':
    runway.run(debug=True, model_options={
        'iharm_checkpoint': 'checkpoints/hrnet18_idih256.pth',
        'unet_checkpoint': 'checkpoints/cp_MaskAttentionUNet_epoch_10.pth',
    })
