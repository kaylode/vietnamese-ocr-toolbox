import torch


class NormMaxMin:
    def __call__(self, x):
        m = torch.min(x)
        M = torch.max(x)
        return (x.float() - m) / (M - m)
