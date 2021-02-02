from torch import nn


def siz(x, n):
    print(n, list(x.size()))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class linear(nn.Module):
    def __init__(self):
        size_hidden = 256
        dropout = 0.1
        super(linear, self).__init__()
        self.linear1  = nn.Linear(784, size_hidden)
        self.drop     = nn.Dropout(dropout)
        self.linear2  = nn.Linear(size_hidden, 784)
        self.relu     = nn.ReLU()
        self.flatten  = Flatten()
        self.do_flatten = True
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x


class cnn_v1(nn.Module):
    def __init__(self):
        super(cnn_v1, self).__init__()
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


class cnn_v2(nn.Module):
    def __init__(self):
        super(cnn_v2, self).__init__()
        # self.name = "cnn v2"
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


class cnn_v3(nn.Module):
    def __init__(self):
        super(cnn_v3, self).__init__()
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
    
    def sigmoid(self, x):
        return nn.functional.sigmoid(x)

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
        x = self.sigmoid(x)
        return x
