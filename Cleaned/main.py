import helper_functions as hf
hf.lineprint("Loading libraries...")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn
import autoencoder_functions as ac
import warnings
import seaborn as sns
import json
from architectures import chosen_cnn_autoencoder as cnn_architecture
from architectures import chosen_lnn_autoencoder as linear_architecture

hf.lineprint("Loading datasets and parameters...")
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style('darkgrid')
sns.set_context('talk')
sns.set(font_scale=0.7)
hf.myprint("Completed")
torch.manual_seed(12345678)

use_gpu = torch.cuda.is_available() and True
global_device = torch.device("cuda:0" if use_gpu else "cpu")

try:
    hf.myprint(f"Running on {torch.cuda.get_device_name(0)}")
except Exception:
    hf.myprint("No GPU available")

mnist_mean = 0.1307
mnist_standard_deviation = 0.3081

do_normalization = False

if do_normalization:
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
else:
    mnist_data = datasets.MNIST('./data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()]))

    mnist_test = datasets.MNIST('./data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()]))


# print(mnist_test[6][0])

def train_cnn(number_of_epochs):
    return ac.train_for_assigment(
        "Convolutional Autoencoder",
        get_model(1),
        number_of_epochs,
        mnist_data, mnist_test)

def train_linear(number_of_epochs):
    return ac.train_for_assigment(
        "Linear Autoencoder",
        get_model(0),
        number_of_epochs,
        mnist_data, mnist_test)

def get_model(mode=1):
    if (mode == 0):
        model = linear_architecture()
    else:
        model = cnn_architecture()
    if use_gpu:
        model.cuda()
    return model


def train_and_save(epochs):
    training_results = train_cnn(epochs)
    name = training_results["autoencoder"].name
    norm_str = "normalized" if do_normalization else "unnormalized"
    basename = f"{name}_{epochs}epochs_{norm_str}_bce"
    hf.save_training_results(basename, training_results)
    hf.myprint("Success")

def main():
    train_and_save(30)

if __name__ == "__main__":
    # pass
    main()