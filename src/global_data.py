import torch
import torchvision
import torchvision.datasets as datasets

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