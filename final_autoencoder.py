import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
import global_var as glval
from torch import nn
from architectures import chosen_cnn_autoencoder as cnn_aut 
from architectures import chosen_lnn_autoencoder as ln_aut

def testEncoder(an_encoder, test_loader, train_loader):
    hf.myprint("Testing encoder...")
    an_encoder.eval()
    loss_func1 = nn.MSELoss()
    loss_func2 = nn.BCELoss()
    test_loss1  = 0
    test_loss2  = 0
    train_loss1 = 0
    train_loss2 = 0
    with torch.no_grad():
        for data, _ in test_loader:
            output = an_encoder(data)
            test_loss1 += loss_func1(output, data).item()
            test_loss2 += loss_func2(output, data).item()
        for data, _ in train_loader:
            output = an_encoder(data)
            train_loss1 += loss_func1(output, data).item()
            train_loss2 += loss_func2(output, data).item()
    train_loss1 /= len(train_loader)
    train_loss2 /= len(train_loader)
    test_loss1 /= len(test_loader)
    test_loss2 /= len(test_loader)
    return (train_loss1, train_loss2, test_loss1, test_loss2)

def trainEpoch(epoch, a_encoder, optimizer, lossFunc, train_loader, train_losses_av):
    a_encoder.train()
    train_loss_av = 0
    for batch_n, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        output = a_encoder(data)
        loss = lossFunc(output, data)
        train_loss_av += loss.detach().item()
        loss.backward()
        optimizer.step()
        if (batch_n % 5 == 0):
            hf.myprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_n * len(data), len(train_loader.dataset),
                100. * batch_n / len(train_loader), loss.item()))
    train_loss_av /= len(train_loader)
    hf.myprint(f"Train loss Average: {train_loss_av}")
    train_losses_av.append(train_loss_av)

def train_cnn(n_epochs, mnist_data, mnist_test):
    optimizerFunc = torch.optim.Adam
    learning_rate = 0.001
    lossFunc = nn.BCELoss()
    batch_size = 1000
    a_encoder = cnn_aut()
    hf.myprint(f"Training Convolutionl autoencoder with {n_epochs} epochs")
    hf.myprint(f"Using Binary Cross Entropy as loss function. Adam as optimizer with learning rate: {learning_rate}")
    dataset_full = torch.utils.data.ConcatDataset([mnist_data,mnist_test])
    new_train, new_test = torch.utils.data.random_split(dataset_full, [50000,20000])

    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
    optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate)
    train_losses_av = []
    train_error_arr1 = []
    train_error_arr2 = []
    test_error_arr1 = [] 
    test_error_arr2 = [] 
    test_results = testEncoder(a_encoder, test_loader, train_loader)
    train_error_arr1.append(test_results[0])
    train_error_arr2.append(test_results[1])
    test_error_arr1.append(test_results[2])
    test_error_arr2.append(test_results[3])
    hf.myprint(f"Starting Train MSELoss: {test_results[0]} \n\tStarting Test MSELoss: {test_results[2]}")
    hf.myprint(f"Starting Train BCELoss: {test_results[1]} \n\tStarting Test BCELoss: {test_results[3]}")
    
    for epoch in range(n_epochs):
        optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate)
        trainEpoch(epoch, a_encoder, optimizer, lossFunc, train_loader, train_losses_av)
        test_results = testEncoder(a_encoder, test_loader, train_loader)
        train_error_arr1.append(test_results[0])
        train_error_arr2.append(test_results[1])
        test_error_arr1.append(test_results[2])
        test_error_arr2.append(test_results[3])
        hf.myprint(f"Current av Train MSELoss: {test_results[0]} \n\tCurrent av Test MSELoss: {test_results[2]}")
        hf.myprint(f"Current av Train BCELoss: {test_results[1]} \n\tCurrent av Test BCELoss: {test_results[3]}")
    
    return_dict = {
        "autoencoder": a_encoder,
        "train_losses_av": train_losses_av,
        "train_error_arr1": train_error_arr1,
        "train_error_arr2": train_error_arr2,
        "test_error_arr1": test_error_arr1,
        "test_error_arr2": test_error_arr2
    }
    return return_dict
