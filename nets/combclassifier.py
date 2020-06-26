from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import *
import numpy as np
import copy


class CombinatorialClassifier(nn.Module):
    partition_weight = None

    def __init__(self, num_classes, num_partitionings, num_partitions, feature_dim, additive=False):
        super(CombinatorialClassifier, self).__init__()
        self.classifiers = nn.Linear(feature_dim, num_partitions * num_partitionings)
        self.num_classes = num_classes
        self.num_partitionings = num_partitionings
        self.num_partitions = num_partitions
        #Adds a persistent buffer to the module.
        #This is typically used to register a buffer that should not to be considered a model parameter.
        #For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.

        self.register_buffer('partitionings', -torch.ones(num_partitionings, num_classes).long())

        self.additive = additive

    def set_partitionings(self, partitionings_map):
        self.partitionings.copy_(torch.LongTensor(partitionings_map).t())
        arange = torch.arange(self.num_partitionings).view(-1, 1).type_as(self.partitionings)
        #arange를 더해준다.? -> 01110, 23332
        self.partitionings.add_(arange * self.num_partitions)

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            if self.partition_weight is None:
                params.grad.mul_(self.num_partitionings)
            else:
                params.grad.mul_(self.partition_weight.sum())

    def forward(self, input, weight=None, output_sum=True, return_meta_dist=False, with_feat=False):
        assert self.partitionings.sum() > 0, 'Partitionings is never given to the module.'

        all_output = self.classifiers(input)
        all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)
        all_output = F.log_softmax(all_output, dim=2)
        if return_meta_dist:
            return all_output
        all_output = all_output.view(-1, self.num_partitionings * self.num_partitions)
        output = all_output.index_select(1, self.partitionings.view(-1))
        output = output.view(-1, self.num_partitionings, self.num_classes)

        if weight is not None:
            weight = weight.view(1, -1, 1)
            output = output * weight
            self.partition_weight = weight

        if output_sum:
            output = output.sum(1)

        if with_feat:
            return input, output
        return output


class EnsembleClassifier(nn.Module):
    def __init__(self, num_classes, num_ensembles, feature_dim, additive=False):
        super(EnsembleClassifier, self).__init__()
        self.classifiers = nn.Linear(feature_dim, num_classes * num_ensembles)
        self.num_classes = num_classes
        self.num_ensembles = num_ensembles

        self.additive = additive

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            params.grad.mul_(self.num_ensembles)

    def forward(self, input, weight=None):
        all_output = self.classifiers(input)
        if self.additive and False:
            raise NotImplementedError

            all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)
            all_output = F.softmax(all_output, dim=2).view(-1, self.num_partitionings * self.num_partitions)
            output = all_output.index_select(1, self.partitionings.view(-1))

            output = output.view(-1, self.num_partitionings, self.num_classes)
            _sum = output.sum(dim=2, keepdim=True)
            output /= _sum.detach()

            # output = all_output.index_select(1, self.partitionings.view(-1))
            # output = F.softmax(output.view(-1, self.num_partitionings, self.num_classes), dim=236_comb_fromZeroNoise)
            if weight is None:
                output = output.sum(1) / self.num_partitionings
            else:
                weight = weight.view(1, -1, 1)
                output = output * weight
                output = output.sum(1)

            output = torch.log(output)
        else:
            all_output = all_output.view(-1, self.num_ensembles, self.num_classes)
            output = F.log_softmax(all_output, dim=2).sum(1)

        return output
