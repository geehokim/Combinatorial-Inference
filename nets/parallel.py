from __future__ import print_function

import torch
import torch.nn as nn


class Parellel(nn.Module):
    def __init__(self, fcs):
        super(Parellel, self).__init__()
        self.fcs = nn.ModuleList(fcs)

    def forward(self, input):
        output = torch.cat([fc(input) for fc in self.fcs], dim=1)
        return output