from torch import nn, Tensor


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x: Tensor):
        return x.reshape(self.shape)
