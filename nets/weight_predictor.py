from __future__ import print_function

import torch
import torch.nn as nn


class WeightPredictor(nn.Module):
    def __init__(self, num_classes, num_partitions, hidden_dim=1024):
        super(WeightPredictor, self).__init__()
        self.h = nn.Parameter(torch.zeros(1, hidden_dim))
        self.predictor = nn.Linear(hidden_dim, num_classes*num_partitions)

        self.num_classes = num_classes
        self.num_partitions = num_partitions

    def forward(self, input):
        normalized_weight = torch.sigmoid(self.predictor(self.h).view(1, self.num_partitions, self.num_classes))

        output = (input * normalized_weight).sum(1)
        norm = torch.log(torch.exp(output).sum(1, keepdim=True)+1e-20)

        output = output - norm
        return output


class WeightPredictorFromFeature(nn.Module):
    def __init__(self, num_classes, num_partitions, hidden_dim=2048):
        super(WeightPredictorFromFeature, self).__init__()
        self.predictor = nn.Linear(hidden_dim, num_classes*num_partitions)

        self.num_classes = num_classes
        self.num_partitions = num_partitions

    def forward(self, input):
        feature = input[0]
        probs = input[1]

        normalized_weight = torch.sigmoid(self.predictor(feature).view(-1, self.num_partitions, self.num_classes))

        output = (probs * normalized_weight).sum(1)
        norm = torch.log(torch.exp(output).sum(1, keepdim=True)+1e-20)

        output = output - norm
        return output