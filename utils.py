import statistics
import torch
import torch.nn.functional as F

def load_partial_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class MovingAverage(object):
    def __init__(self, n=100):
        self.sum = 0
        self.n = n
        self.buffer = []
    
    def update(self, v):
        if len(self.buffer) == self.n:
            self.buffer.pop(0)
        self.buffer.append(v)

    def get(self):
        return round(statistics.mean(self.buffer), 5)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
