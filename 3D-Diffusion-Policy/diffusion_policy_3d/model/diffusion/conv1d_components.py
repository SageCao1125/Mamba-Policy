import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange
from mamba_ssm import Mamba, Mamba2
# from mamba_ssm.modules.mamba_simple import Mamba
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)
sys.path.append(script_dir)
from hydra_ssm import Hydra
# from mamba_ssm.modules.mamba_simple import Mamba

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class MambaBlock_v1(nn.Module):
    '''
        Mamba --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size=None, n_groups=8, use_bimamba=False):
        super().__init__()
        assert inp_channels == out_channels

        if use_bimamba:
            self.block = nn.Sequential(
                Mamba(d_model=inp_channels,expand=2,bimamba=use_bimamba),
            )
        else:
            self.block = nn.Sequential(
                Mamba(d_model=inp_channels,expand=2),
            )
        self.norm = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        '''
        x : [ batch_size x in_channels x horizon ]
        '''
        x = x.permute(0, 2, 1) # [ batch_size x horizon x in_channels ]
        x = self.block(x)
        x = x.permute(0, 2, 1) # [ batch_size x in_channels x horizon ]
        x = self.norm(x)
        return x

class MambaBlock_v2(nn.Module):
    '''
        Mamba --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size=None, n_groups=8):
        super().__init__()
        assert inp_channels == out_channels
        self.block = nn.Sequential(
                Mamba2(d_model=inp_channels, headdim=32),
            )
        self.norm = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        '''
        x : [ batch_size x in_channels x horizon ]
        '''
        x = x.permute(0, 2, 1) # [ batch_size x horizon x in_channels ]
        x = self.block(x)
        x = x.permute(0, 2, 1) # [ batch_size x in_channels x horizon ]
        x = self.norm(x)

        return x

class MambaBlock_hydra(nn.Module):
    '''
        Mamba --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size=None, n_groups=8):
        super().__init__()
        assert inp_channels == out_channels
        self.block = nn.Sequential(
                Hydra(d_model=inp_channels),
            )
        self.norm = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        '''
        x : [ batch_size x in_channels x horizon ]
        '''
        x = x.permute(0, 2, 1) # [ batch_size x horizon x in_channels ]
        x = self.block(x)
        x = x.permute(0, 2, 1) # [ batch_size x in_channels x horizon ]
        x = self.norm(x)

        return x

def test():
    # cb = Conv1dBlock(16, 128, kernel_size=3).cuda()
    cb = MambaBlock_v1(128, 128, n_groups=8).cuda()
    cb = MambaBlock_v2(128, 128, n_groups=8).cuda()
    cb = MambaBlock_hydra(128, 128, n_groups=8).cuda()
    x = torch.zeros((1,128,4)).cuda()
    o = cb(x)
    print(o.shape)


if __name__ == '__main__':
    test()

    # x = torch.zeros((1,10,32)).cuda()
    # from mamba_ssm.modules.mamba_simple import Mamba
    # from mamba_ssm.modules.mamba_simple import Mamba
    # mamba = Mamba(d_model=32,expand=2,bimamba=True).cuda()

    # y = mamba(x)
    # print(y.shape)
    # print(y.grad)