import statistics
import torch.nn.functional as F

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
        return statistics.mean(self.buffer)
