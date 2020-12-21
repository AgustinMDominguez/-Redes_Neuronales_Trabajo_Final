# Agustin Marcelo Dominguez - Div 2020

import helper_functions as hf
hf.lineprint("Loading libraries...")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn
import autoencoder as ac
import global_var as glval
import warnings
import seaborn as sns
import json

from final_autoencoder import train_cnn

hf.lineprint("Loading datasets and parameters...")
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style('darkgrid')
sns.set_context('talk')
sns.set(font_scale=0.7)
hf.myprint("Completed")
torch.manual_seed(12345678)

try:
    hf.myprint(f"Running on {torch.cuda.get_device_name(0)}")
except Exception:
    hf.myprint("No GPU available")

mnist_mean = 0.1307
mnist_standard_deviation = 0.3081

mnist_data = datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), 
                               torchvision.transforms.Normalize((mnist_mean,), (mnist_standard_deviation,))
                               ]))

mnist_test = datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), 
                               torchvision.transforms.Normalize((mnist_mean,), (mnist_standard_deviation,))
                               ]))

epochs = 4

def runBestCNNAutoencoder():
    hf.lineprint("Processing Convolutional Autoencoder...")
    print(train_cnn(3, mnist_data, mnist_test))

def runBestLNNAutoencoder():
    hf.lineprint("Processing Liner Autoencoder...")

runBestCNNAutoencoder()