import numpy as np
print(np.__version__)
import math

import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet(nn.Module):
    def __init__(self, MSA, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(

            #input channels should be = self.MSA.nbr_sequences
            nn.Conv2d(4, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            # might need to remove this check vid @ 1:45:48
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            # shady input size to checK
            #nn.Linear(32 * MSA.nbr_sequences * MSA.max_length, MSA.nbr_sequences * MSA.max_length)
            nn.Linear(32 * MSA.max_length * (len(MSA.sequence_constructor.alphabet) +1 ), MSA.nbr_sequences * MSA.max_length)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),

            # shady input size to checK
            #nn.Linear(2 * MSA.nbr_sequences * MSA.max_length, 1),
            nn.Linear(3 * MSA.max_length * (len(MSA.sequence_constructor.alphabet) +1 ), 1),
            #nn.PReLU()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x