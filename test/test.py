from __future__ import print_function
from __future__ import division

from abc import ABC

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
from torch.utils.data import DataLoader
import random
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.autograd import Variable

criterion = nn.CrossEntropyLoss()

output = Variable(torch.randn(10, 120).float())
target = Variable(torch.FloatTensor(10).uniform_(0, 120).long())

loss = criterion(output, target)