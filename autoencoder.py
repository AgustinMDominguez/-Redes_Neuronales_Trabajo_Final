import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
import global_var as glval
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def siz(x, n):
    print(n, list(x.size()))

class cnn_aut_v1(nn.Module):
    def __init__(self):
        super(cnn_aut_v1, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(  1,   8, 7)
        self.conv2 = nn.Conv2d(  8,  16, 3)
        self.conv3 = nn.Conv2d( 16,  64, 3)
        self.conv4 = nn.Conv2d( 64,  16, 3)
        self.conv5 = nn.Conv2d( 16,   8, 3)
        self.conv6 = nn.Conv2d(  8,   1, 3)

    def interpolate(self, x, size_x, size_y):
        return nn.functional.interpolate(x, size=(size_x,size_y), mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.interpolate(x, 16, 16)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.interpolate(x, 24, 24)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.interpolate(x, 30, 30)
        x = self.conv6(x)
        x = self.relu(x)
        return x

class cnn_aut_v2(nn.Module):
    def __init__(self):
        super(cnn_aut_v2, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(  1,  20, 5)
        self.conv2 = nn.Conv2d( 20,  64, 3)
        self.conv3 = nn.Conv2d( 64,  32, 3)
        self.conv4 = nn.Conv2d( 32,  16, 3)
        self.conv5 = nn.Conv2d( 16,   1, 3)

    def interpolate(self, x, size_x, size_y):
        return nn.functional.interpolate(x, size=(size_x,size_y), mode='bilinear')

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        #Decoder
        x = self.interpolate(x, 12, 12)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.interpolate(x, 22, 22)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.interpolate(x, 30, 30)
        x = self.conv5(x)
        x = self.relu(x)
        return x

class cnn_aut_v3(nn.Module):
    def __init__(self):
        super(cnn_aut_v3, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(  1,  32, 5)
        self.conv2 = nn.Conv2d( 32,  64, 3)
        self.conv3 = nn.Conv2d( 64,  32, 3)
        self.conv4 = nn.Conv2d( 32,  16, 3)
        self.conv5 = nn.Conv2d( 16,  8, 3)
        self.conv6 = nn.Conv2d(  8,  1, 3)

    def interpolate(self, x, size_x, size_y):
        return nn.functional.interpolate(x, size=(size_x,size_y), mode='bilinear')

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        #Decoder
        x = self.interpolate(x, 16, 16)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.interpolate(x, 26, 26)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.interpolate(x, 30, 30)
        x = self.conv6(x)
        x = self.relu(x)
        return x

autoencoder = cnn_aut_v3

def train_epoch(epoch, autoenc, optimizer, lossFunc, train_loader,
        log, log_interval, train_losses_drop):
    autoenc.train()
    train_loss_drop = 0
    for batch_n, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        output = autoenc(data)
        loss = lossFunc(output, data)
        train_loss_drop += loss.detach().item()
        loss.backward()
        optimizer.step()
        if (log and batch_n % log_interval == 0):
            hf.myprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_n * len(data), len(train_loader.dataset),
                100. * batch_n / len(train_loader), loss.item()))
    train_loss_drop /= len(train_loader)
    hf.myprint(f"Train loss Drop: {train_loss_drop}")
    train_losses_drop.append(train_loss_drop)

def testEncoder(autoenc, test_loader, train_loader, lossFunc):
    autoenc.eval()
    test_loss  = 0
    train_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            output = autoenc(data)
            test_loss += lossFunc(output, data).item()
        for data, _ in train_loader:
            output = autoenc(data)
            train_loss += lossFunc(output, data).item()
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    return (train_loss, test_loss)

def get_learning_rate_update(epoch, learning_rate):
    if(epoch%10==0 and epoch!=0):
        new_learning_rate = learning_rate/1.5
        hf.myprint(f"Lowering learning rate from {learning_rate} to {new_learning_rate}")
        return new_learning_rate
    else:
        return learning_rate

def train_cnn(
    mnist_data,
    mnist_test,
    saved_encoder  = None,
    n_epochs       = glval.n_epochs,
    optimizerFunc  = glval.optimizerFunc,
    lossFunc       = glval.lossFunc,
    learning_rate  = glval.learning_rate,
    update_lr      = False,
    has_momentum   = glval.has_momentum,
    momentum       = glval.momentum,
    batch_size     = glval.batch_size,
    dropout_rate   = glval.dropout_rate,
    log            = True,
    log_interval   = glval.log_interval):
    hf.myprint(f"Training Convolutionl autoencoder with {n_epochs} epochs")
    mom = 0 if not has_momentum else momentum
    hf.myprint(f"learning rate: {learning_rate} - dropout rate: {dropout_rate} - momentum: {mom}")
    hf.myprint(f"Testing dataset size:{len(mnist_test)} - Training dataset size:{len(mnist_data)}")
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
    if saved_encoder == None:
        a_encoder = autoencoder()
    else:
        hf.myprint("Loaded saved network")
        a_encoder = saved_encoder
    if has_momentum:
        optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate)
    train_losses_drop = []
    train_error_arr = []
    test_error_arr = [] 
    test_results = testEncoder(a_encoder, test_loader, train_loader, lossFunc)
    train_error_arr.append(test_results[0])
    test_error_arr.append(test_results[1])
    if log:
        hf.myprint(f"Starting Train Loss: {test_results[0]} \n\tStarting Test  Loss: {test_results[1]}")
    for epoch in range(n_epochs):
        learning_rate = get_learning_rate_update(epoch, learning_rate)
        if update_lr:
            if has_momentum:
                optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate, momentum=momentum)
            else:
                optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate)
        train_epoch(epoch, a_encoder, optimizer, lossFunc, train_loader, log, log_interval, train_losses_drop)
        test_results = testEncoder(a_encoder, test_loader, train_loader, lossFunc)
        train_error_arr.append(test_results[0])
        test_error_arr.append(test_results[1])
        if log:
            hf.myprint(f"Current av Train Loss: {test_results[0]} \n\tCurrent av Test  Loss: {test_results[1]}")
    return_dict = {
        "autoencoder": a_encoder,
        "train_losses_drop": train_losses_drop,
        "train_error_arr": train_error_arr,
        "test_error_arr": test_error_arr
    }
    return return_dict

def compareEncoder(autoenc, mnist_test, number_range, save=False, basename='', tight=False):
    flatten = Flatten()
    if type(number_range) == type(tuple):
        rang = range(number_range[0], number_range[1])
    else:
        rang = number_range
    for i in rang:
        _, axes = plt.subplots(figsize=(10, 6), ncols=2, nrows=1)
        axes[0].imshow(mnist_test[i][0][0], cmap='gray')
        img = autoenc(flatten(mnist_test[i][0]).unsqueeze(0)).detach().numpy().reshape([28,28])
        axes[1].imshow(img, cmap = 'gray',)
        if save:
            filename = basename+f'_comp{i}.png'
            if tight:
                plt.axis('off') #doesn't work
                plt.savefig( filename,dpi=150, bbox_inches='tight', transparent="True", pad_inches=0)
            else:
                plt.savefig(filename, dpi=200)
            
            hf.myprint("\tSaved "+filename)
            plt.clf()
        else:
            plt.show()

def train_cnn_for_assigment(n_epochs, mnist_data, mnist_test):
    optimizerFunc = torch.optim.Adam
    has_momentum = False
    momentum = None
    learning_rate = 0.001
    update_lr = False
    lossFunc = nn.MSELoss()
    hf.myprint("Using MSE as loss function")
    dataset_full = torch.utils.data.ConcatDataset([mnist_data,mnist_test])
    new_train, new_test = torch.utils.data.random_split(dataset_full, [50000,20000])
    return train_cnn(
        new_train, new_test, saved_encoder=None,
        n_epochs=n_epochs, optimizerFunc=optimizerFunc, lossFunc=lossFunc,
        learning_rate=learning_rate, update_lr=update_lr, has_momentum=has_momentum, momentum=momentum, batch_size=1000,
        dropout_rate=0.1, log=True, log_interval=10
    )
