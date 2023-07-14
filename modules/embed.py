import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import random
import numpy as np
seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
'''
Reshaping:
Assuming that the data takes the form (Batch, Time Series Length, Time Series Dimension) -> (16,100,10)
And assuming that we use seg_len = 20, d_model = 512

[1]: ([16, 100, 20]) | original input
[2]: ([1600, 20]) | Isolating the segment, so that we can embed it. 
[3]: ([1600, 512]) | The size of the embedded segment
[4]: ([16, 20, 5, 512]) | 16 examples, each with 20 dimensions, where each dimension has 5 embedded segments with 512 features. (5 because ts length / segment length = 5)
'''


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        # [1]
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        # [2]
        x_embed = self.linear(x_segment)
        # [3]
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        # [4]
        return x_embed