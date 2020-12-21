from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class linear_autoencoder_v1(nn.Module):
    def __init__(self):
        super(linear_autoencoder_v1, self).__init__()
        size_hidden = 256
        dropout = 0.1
        self.name = "linear_v1"
        self.linear1  = nn.Linear(784, size_hidden)
        self.drop     = nn.Dropout(dropout)
        self.linear2  = nn.Linear(size_hidden, 784)
        self.relu     = nn.ReLU()
        self.flatten  = Flatten()
        self.do_flatten = True
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

def siz(x, n):
    print(n, list(x.size()))

class cnn_aut_v1(nn.Module):
    def __init__(self):
        super(cnn_aut_v1, self).__init__()
        self.name = "cnn_v1"
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(  1,   8, 7)
        self.conv2 = nn.Conv2d(  8,  16, 3)
        self.conv3 = nn.Conv2d( 16,  64, 3)
        self.conv4 = nn.Conv2d( 64,  16, 3)
        self.conv5 = nn.Conv2d( 16,   8, 3)
        self.conv6 = nn.Conv2d(  8,   1, 3)
        self.do_flatten = False

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
        self.name = "cnn_v2"
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.do_flatten = False
        self.name 
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
        self.name = "cnn_v3"
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.do_flatten = False
        self.conv1 = nn.Conv2d( 1, 10, 3)
        self.conv2 = nn.Conv2d(10, 30, 7)
        self.conv3 = nn.Conv2d(30, 10, 3)
        self.conv4 = nn.Conv2d(10, 1, 3)

    def interpolate(self, x, size_x, size_y):
        return nn.functional.interpolate(x, size=(size_x,size_y), mode='bilinear')
    
    def sigmoid(self, x):
        return nn.functional.sigmoid(x)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.interpolate(x, 16, 16)
        x = self.relu(self.conv3(x))
        x = self.interpolate(x, 30, 30)
        x = self.relu(self.conv4(x))
        return x

class cnn_aut_v4(nn.Module):
    def __init__(self):
        super(cnn_aut_v4, self).__init__()
        self.name = "cnn_v4"
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.do_flatten = False
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.conv2 = nn.Conv2d(20, 1, 3)
        self.upsample1 = nn.Upsample((30,30), mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def interpolate(self, x, size_x, size_y):
        return nn.functional.interpolate(x, size=(size_x,size_y), mode='bilinear')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.upsample1(x)
        x = self.sigmoid(self.conv2(x))
        return x

class cnn_aut_v5(nn.Module):
    def __init__(self):
        super(cnn_aut_v5, self).__init__()
        self.name = "cnn_v5"
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.do_flatten = False
        
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        self.sigmoid = nn.Sigmoid()
        self.t_conv1 = nn.ConvTranspose2d(64,32,7)
        self.t_conv2 = nn.ConvTranspose2d(32,16,3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(16,1,3, stride=2, padding=1, output_padding=1)

    def interpolate(self, x, size_x, size_y):
        return nn.functional.interpolate(x, size=(size_x,size_y), mode='bilinear')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        x = self.sigmoid(self.t_conv3(x))
        return x

class cnn_aut_v6(nn.Module):
    def __init__(self):
        super(cnn_aut_v6, self).__init__()
        self.name = "cnn_v6"
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.do_flatten = False
        
        self.conv1 = nn.Conv2d(1, 20, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(20, 50, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(50, 100, 7)
        self.sigmoid = nn.Sigmoid()
        self.t_conv1 = nn.ConvTranspose2d(100,50,7)
        self.t_conv2 = nn.ConvTranspose2d(50,20,3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(20,1,3, stride=2, padding=1, output_padding=1)

    def interpolate(self, x, size_x, size_y):
        return nn.functional.interpolate(x, size=(size_x,size_y), mode='bilinear')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        x = self.sigmoid(self.t_conv3(x))
        return x

chosen_cnn_autoencoder = cnn_aut_v6
chosen_lnn_autoencoder = linear_autoencoder_v1