from torch import nn, Tensor


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.dims = args

    def forward(self, x: Tensor):
        return x.permute(self.dims)
