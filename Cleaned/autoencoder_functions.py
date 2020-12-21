import helper_functions as hf
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn
from architectures import Flatten

use_gpu = torch.cuda.is_available() and True
global_device = torch.device("cuda:0" if use_gpu else "cpu")

def get_loss(autoencoder, data, lossFunc):
    output = autoencoder(data)
    if autoencoder.do_flatten:
        return lossFunc(output, autoencoder.flatten(data))
    else:
        return lossFunc(output, data)

def testEncoder(autoenc, test_loader, train_loader, lossFunc):
    autoenc.eval()
    test_loss  = 0
    train_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(global_device)
            test_loss += get_loss(autoenc, data, lossFunc).item()
        for data, _ in train_loader:
            data = data.to(global_device)
            train_loss += get_loss(autoenc, data, lossFunc).item()
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    return (train_loss, test_loss)

def train_epoch(epoch, autoenc, optimizer, lossFunc, train_loader, train_losses_av):
    autoenc.train()
    train_loss_av = 0
    log_interval = 10
    for batch_n, (data, _) in enumerate(train_loader):
        data = data.to(global_device)
        optimizer.zero_grad()
        output = autoenc(data)
        loss = get_loss(autoenc, data, lossFunc)
        train_loss_av += loss.detach().item()
        loss.backward()
        optimizer.step()
        if (batch_n % log_interval == 0):
            hf.myprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_n * len(data), len(train_loader.dataset),
                100. * batch_n / len(train_loader), loss.item()))
    train_loss_av /= len(train_loader)
    hf.myprint(f"Train loss average: {train_loss_av}")
    train_losses_av.append(train_loss_av)

def train_for_assigment(ac_name, a_encoder, n_epochs, mnist_data, mnist_test):
    hf.myprint(f"Starting assigment for {ac_name}...")
    optimizerFunc = torch.optim.Adam
    learning_rate = 0.001
    optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate)
    batch_size = 250
    # lossFunc = nn.MSELoss()
    lossFunc = nn.BCELoss()
    dataset_full = torch.utils.data.ConcatDataset([mnist_data,mnist_test])
    mnist_data, mnist_test = torch.utils.data.random_split(dataset_full, [50000,20000])
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
    train_losses_av = []
    train_error_arr = []
    test_error_arr = [] 
    test_results = testEncoder(a_encoder, test_loader, train_loader, lossFunc)
    hf.myprint(f"Training {ac_name} with {n_epochs} epochs. Using MSE as loss function and Adam as optimizer with learning rate: {learning_rate}")
    hf.myprint(f"Starting Train Loss: {test_results[0]} \n\tStarting Test  Loss: {test_results[1]}")
    for epoch in range(n_epochs):
        train_epoch(epoch, a_encoder, optimizer, lossFunc, train_loader, train_losses_av)
        test_results = testEncoder(a_encoder, test_loader, train_loader, lossFunc)
        train_error_arr.append(test_results[0])
        test_error_arr.append(test_results[1])
        hf.myprint(f"Current av Train Loss: {test_results[0]} \n\tCurrent av Test  Loss: {test_results[1]}")
    return_dict = {
        "autoencoder": a_encoder,
        "train_losses_av": train_losses_av,
        "train_error_arr": train_error_arr,
        "test_error_arr": test_error_arr
    }
    return return_dict
