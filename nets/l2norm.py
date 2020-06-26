from __future__ import print_function

import torch
import torch.nn as nn


class L2NormLayer(nn.Module):
    def __init__(self, scale=True):
        super(L2NormLayer, self).__init__()
        self.scale = scale
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, input):
        output = self.l2_norm(input)
        if self.scale:
            output = output * self.s
        return output

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output