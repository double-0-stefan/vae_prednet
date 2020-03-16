import os
import math
from torch import nn, max
import logging
import urllib.request as request
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torchvision.transforms.functional as TTF
from torch import FloatTensor
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import scipy.io as sio
from numbers import Number
import numpy as np
import PIL
from itertools import chain, product
from collections.abc import Iterable
from torch.utils.data import TensorDataset
from utils import dist 
#from utils import blur_utils
import time 
import torch
import glob
import torch
import kornia.filters as kf

#from animalai.envs import UnityEnvironment
#from animalai.envs.arena_config import ArenaConfig
#from data.MovingMNIST import MovingMNIST
from torchvision.utils import save_image



class Gaussian_Smooth(object):
    def __init__(self):
		    self.f = kf.GaussianBlur2d((15,15), (7,7))
		    #self.s = transforms.Grayscale(num_output_channels=1)

    def __call__(self, sample):
        return self.f(sample)#.view(1,3,96,96)).view(1,96,96)
		