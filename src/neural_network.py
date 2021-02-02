import torch
from torch import nn
from src.global_data import mnist_data, mnist_test
import src.helper_functions as hf

class NeuralNetwork():
    def __init__(self, architecture, lossfunc=nn.BCELoss(), batch_size=1000, do_random_split=False):
        self.architecture = architecture
        if do_random_split:
            dataset_full = torch.utils.data.ConcatDataset([mnist_data,mnist_test])
            new_train, new_test = torch.utils.data.random_split(dataset_full, [50000,20000])
            self.mnist_data = new_train
            self.mnist_test = new_test
        else:
            self.mnist_data = mnist_data
            self.mnist_test = mnist_test
        self.train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
        self.test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
        self.lossfunc = lossfunc

    def set_loss_function(self, lossfunc):
        self.lossfunc = lossfunc

    def init_encoder(self):
        self.encoder = self.architecture()
        self.train_losses_av = []
        self.train_error_arr1 = []
        self.train_error_arr2 = []
        self.test_error_arr1 = [] 
        self.test_error_arr2 = [] 

    def get_early_performance(self):
        self.init_encoder()

    def test_encoder(self):
        hf.myprint("Testing encoder...")
        self.encoder.eval()
        loss_func1 = nn.MSELoss()
        loss_func2 = nn.BCELoss()
        test_loss1  = 0
        test_loss2  = 0
        train_loss1 = 0
        train_loss2 = 0
        with torch.no_grad():
            for data, _ in self.test_loader:
                output = self.encoder(data)
                test_loss1 += loss_func1(output, data).item()
                test_loss2 += loss_func2(output, data).item()
            for data, _ in self.train_loader:
                output = self.encoder(data)
                train_loss1 += loss_func1(output, data).item()
                train_loss2 += loss_func2(output, data).item()
        train_loss1 /= len(self.train_loader)
        train_loss2 /= len(self.train_loader)
        test_loss1 /= len(self.test_loader)
        test_loss2 /= len(self.test_loader)
        return (train_loss1, train_loss2, test_loss1, test_loss2)

    def train_epoch(self, epoch, optimizer):
        self.encoder.train()
        train_loss_av = 0
        for batch_n, (data, _) in enumerate(self.train_loader):
            optimizer.zero_grad()
            output = self.encoder(data)
            loss = self.lossFunc(output, data)
            train_loss_av += loss.detach().item()
            loss.backward()
            optimizer.step()
            if (batch_n % 5 == 0):
                hf.myprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_n * len(data), len(self.train_loader.dataset),
                    100. * batch_n / len(self.train_loader), loss.item()))
        train_loss_av /= len(self.train_loader)
        hf.myprint(f"Train loss Average: {train_loss_av}")
        self.train_losses_av.append(train_loss_av)

    def train_cnn(self, n_epochs):
        optimizerFunc = torch.optim.Adam
        learning_rate = 0.001
        hf.myprint(f"Training Convolutionl autoencoder with {n_epochs} epochs")
        hf.myprint(f"Using Binary Cross Entropy as loss function. Adam as optimizer with learning rate: {learning_rate}")
        optimizer = optimizerFunc(self.encoder.parameters(), lr=learning_rate)
        test_results = self.test_encoder()
        self.train_error_arr1.append(test_results[0])
        self.train_error_arr2.append(test_results[1])
        self.test_error_arr1.append(test_results[2])
        self.test_error_arr2.append(test_results[3])
        hf.myprint(f"Starting Train MSELoss: {test_results[0]} \n\tStarting Test MSELoss: {test_results[2]}")
        hf.myprint(f"Starting Train BCELoss: {test_results[1]} \n\tStarting Test BCELoss: {test_results[3]}")
        
        for epoch in range(n_epochs):
            optimizer = optimizerFunc(self.encoder.parameters(), lr=learning_rate)
            self.train_epoch(epoch, self.encoder, optimizer)
            test_results = self.testEncoder()
            self.train_error_arr1.append(test_results[0])
            self.train_error_arr2.append(test_results[1])
            self.test_error_arr1.append(test_results[2])
            self.test_error_arr2.append(test_results[3])
            hf.myprint(f"Current av Train MSELoss: {test_results[0]} \n\tCurrent av Test MSELoss: {test_results[2]}")
            hf.myprint(f"Current av Train BCELoss: {test_results[1]} \n\tCurrent av Test BCELoss: {test_results[3]}")
