import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import copy
import torchvision.models as models

device = torch.device("cuda:0")

class TemperedSigmoid(nn.Module):
    def __init__(self, s=2, T=2, o=1):
        super().__init__()
        self.s = s
        self.T = T
        self.o = o

    def forward(self, input):
        div = 1 + torch.exp(-1 * self.T *input)
        return self.s / div - self.o

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.grads =[]
        self.grad_dict = {}
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # import ipdb; ipdb.set_trace()
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        output = self.fc2(x)
        return output

class ConvNet_mnist(nn.Module):
    
    def __init__(self, param, num_classes=10):

        super(ConvNet_mnist, self).__init__()
        
        num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        TS = TemperedSigmoid()
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        #out = F.log_softmax(out, dim=1)
        return out

class ConvNet_mnist_TS(nn.Module):
    
    def __init__(self, param, num_classes=10):

        super(ConvNet_mnist_TS, self).__init__()
        
        num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            #nn.ReLU(),
            TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
            TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        #TS = TemperedSigmoid()
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        #out = F.log_softmax(out, dim=1)
        return out

class ConvNet_cifar100(nn.Module):
    
    def __init__(self, param, num_classes=100):
        '''
        dataset = ['mnist', 'cifar_100']
        param = [0, 1, 2, 3, 4] ,   0 : 1w param
                                    1 : 5w param
                                    2 : 10w param
                                    3 : 50w param
                                    4 : 100w param
        '''
        super(ConvNet_cifar100, self).__init__()
        
        num_classes = 100
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(2 * 2 * 128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)


    def forward(self, x):
        TS = TemperedSigmoid()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        #out = TS(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        #out = F.log_softmax(out, dim=1)
        return out



class ResNetResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel=3, downsampling=1, conv_shortcut=False, TS=False, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.kernel, self.downsampling = kernel, downsampling
        self.conv_shortcut= conv_shortcut
        if TS:
            self.activate = TemperedSigmoid()
        else:
            self.activate = nn.ReLU()
        self.shortcut = nn.Conv2d(self.in_channels, filters *4, kernel_size=1,
                      stride=self.downsampling) if self.conv_shortcut else nn.MaxPool2d(kernel_size=1, stride=self.downsampling)

        self.BN_1 = nn.BatchNorm2d(self.in_channels, eps=1.001e-5)
        self.Conv_1 = nn.Conv2d(self.in_channels, filters, kernel_size=1, stride=1, bias=False)
        self.BN_2 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.zeroPad_1 = nn.ZeroPad2d((1,1,1,1))
        self.Conv_2 = nn.Conv2d(filters, filters, kernel_size=self.kernel, stride=self.downsampling, bias=False)
        self.BN_3 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.Conv_3 = nn.Conv2d(filters , filters *4, kernel_size=1)
        
    def forward(self, x):
        x = self.BN_1(x)
        x = self.activate(x)

        residual = self.shortcut(x)

        x = self.Conv_1(x)
        x = self.BN_2(x)
        x = self.activate(x)
        x = self.zeroPad_1(x)
        x = self.Conv_2(x)
        x = self.BN_3(x)
        x = self.activate(x)
        x = self.Conv_3(x)
        x += residual
        return x

class ResNet18v2_cifar10(nn.Module):
    def __init__(self, param, classes=10, *args, **kwargs):
        super().__init__()
        self.classes = classes
        
        '''
        self.zeroPad_1 = nn.ZeroPad2d((3,3,3,3))
        self.Conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False)
        self.zeroPad_2 = nn.ZeroPad2d((1,1,1,1))
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        ##----block
        self.block_1 = ResNetResidualBlock(64, 16, conv_shortcut=True)
        self.block_2 = ResNetResidualBlock(64, 16, downsampling=2)
        self.block_3 = ResNetResidualBlock(64, 32, conv_shortcut=True)
        self.block_4 = ResNetResidualBlock(128, 32, downsampling=2)
        self.block_5 = ResNetResidualBlock(128, 64, conv_shortcut=True)
        self.block_6 = ResNetResidualBlock(256, 64, downsampling=2)
        self.block_7 = ResNetResidualBlock(256, 128, conv_shortcut=True)
        self.block_8 = ResNetResidualBlock(512, 128)
        
        self.BN_1 = nn.BatchNorm2d(512, eps=1.001e-5)
        self.activate = nn.ReLU()
        self.GAP_1 = nn.AvgPool2d(1,1)
        self.fc_1 = nn.Linear(1,classes)
        '''
        self.model = nn.Sequential(
            nn.ZeroPad2d((3,3,3,3)),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.ZeroPad2d((1,1,1,1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResNetResidualBlock(64, 16, conv_shortcut=True),
            ResNetResidualBlock(64, 16, downsampling=2),
            ResNetResidualBlock(64, 32, conv_shortcut=True),
            ResNetResidualBlock(128, 32, downsampling=2),
            ResNetResidualBlock(128, 64, conv_shortcut=True),
            ResNetResidualBlock(256, 64, downsampling=2),
            ResNetResidualBlock(256, 128, conv_shortcut=True),
            ResNetResidualBlock(512, 128),
            nn.BatchNorm2d(512, eps=1.001e-5),
            nn.ReLU(),
            nn.AvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512,classes))
        #self.GAP_1 = nn.AvgPool2d(1)
        #self.fc_1 = nn.Linear(1,classes)
        #self.flatten = nn.Flatten()
    def forward(self, x):
        '''
        x = self.zeroPad_1(x)
        x = self.Conv_1(x)
        x = self.zeroPad_2(x)
        x = self.maxpool_1(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)

        x = self.BN_1(x)
        x = self.activate(x)
        '''
        x = self.model(x)
        #x = self.GAP_1(x)
        #x = self.flatten(x)
        #x = self.fc_1(x)
        return x

class ResNet18v2_cifar10_TS(nn.Module):
    def __init__(self, param, classes=10, *args, **kwargs):
        super().__init__()
        self.classes = classes
        
        '''
        self.zeroPad_1 = nn.ZeroPad2d((3,3,3,3))
        self.Conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False)
        self.zeroPad_2 = nn.ZeroPad2d((1,1,1,1))
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        ##----block
        self.block_1 = ResNetResidualBlock(64, 16, conv_shortcut=True)
        self.block_2 = ResNetResidualBlock(64, 16, downsampling=2)
        self.block_3 = ResNetResidualBlock(64, 32, conv_shortcut=True)
        self.block_4 = ResNetResidualBlock(128, 32, downsampling=2)
        self.block_5 = ResNetResidualBlock(128, 64, conv_shortcut=True)
        self.block_6 = ResNetResidualBlock(256, 64, downsampling=2)
        self.block_7 = ResNetResidualBlock(256, 128, conv_shortcut=True)
        self.block_8 = ResNetResidualBlock(512, 128)
        
        self.BN_1 = nn.BatchNorm2d(512, eps=1.001e-5)
        self.activate = nn.ReLU()
        self.GAP_1 = nn.AvgPool2d(1,1)
        self.fc_1 = nn.Linear(1,classes)
        '''
        self.TS = TemperedSigmoid()
        self.model = nn.Sequential(
            nn.ZeroPad2d((3,3,3,3)),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
            nn.ZeroPad2d((1,1,1,1)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResNetResidualBlock(64, 16, conv_shortcut=True, TS=True),
            ResNetResidualBlock(64, 16, downsampling=2, TS=True),
            ResNetResidualBlock(64, 32, conv_shortcut=True, TS=True),
            ResNetResidualBlock(128, 32, downsampling=2, TS=True),
            ResNetResidualBlock(128, 64, conv_shortcut=True, TS=True),
            ResNetResidualBlock(256, 64, downsampling=2, TS=True),
            ResNetResidualBlock(256, 128, conv_shortcut=True, TS=True),
            ResNetResidualBlock(512, 128, TS=True),
            nn.BatchNorm2d(512, eps=1.001e-5),
            self.TS,
            nn.AvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512,classes))
        #self.GAP_1 = nn.AvgPool2d(1)
        #self.fc_1 = nn.Linear(1,classes)
        #self.flatten = nn.Flatten()
    def forward(self, x):
        '''
        x = self.zeroPad_1(x)
        x = self.Conv_1(x)
        x = self.zeroPad_2(x)
        x = self.maxpool_1(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)

        x = self.BN_1(x)
        x = self.activate(x)
        '''
        x = self.model(x)
        #x = self.GAP_1(x)
        #x = self.flatten(x)
        #x = self.fc_1(x)
        return x

class ConvNet_cifar10_target(nn.Module):
    
    def __init__(self, param, num_classes=10):
        '''
        dataset = ['mnist', 'cifar_100']
        param = [0, 1, 2, 3, 4] ,   0 : 1w param
                                    1 : 5w param
                                    2 : 10w param
                                    3 : 50w param
                                    4 : 100w param
        '''
        super(ConvNet_cifar10_target, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(2, 2))
        self.fc1 = nn.Linear(8 * 8 * 64, 384)
        self.fc2 = nn.Linear(384, 384)
        self.fc3 = nn.Linear(384, 10)
        self.dropout = torch.nn.Dropout(p=0.2)


    def forward(self, x):
        TS = TemperedSigmoid()
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        #out = TS(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        #out = F.log_softmax(out, dim=1)
        return out





class ConvNet_cifar10_shadow(nn.Module):
    
    def __init__(self, param, num_classes=10):
        '''
        dataset = ['mnist', 'cifar_100']
        param = [0, 1, 2, 3, 4] ,   0 : 1w param
                                    1 : 5w param
                                    2 : 10w param
                                    3 : 50w param
                                    4 : 100w param
        '''
        super(ConvNet_cifar10_shadow, self).__init__()
        
        num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #TemperedSigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(2 * 2 * 32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)


    def forward(self, x):
        TS = TemperedSigmoid()
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        #out = TS(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        #out = F.log_softmax(out, dim=1)
        return out



class resnet50(nn.Module):
    
    def __init__(self, param):
        super(resnet50, self).__init__()
        
        
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048,10)
        self.s = nn.Softmax(dim=1)

    def forward(self, x):

        out = self.resnet50(x)
        out = self.s(out)
        return out

class resnet101(nn.Module):
    
    def __init__(self, param):
        super(resnet101, self).__init__()
        
        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Linear(2048,10)
        self.s = nn.Softmax(dim=1)

    def forward(self, x):

        out = self.resnet101(x)
        out = self.s(out)
        return out


class softmax_model(nn.Module):
    
    def __init__(self, n_in, n_out):
        super(softmax_model, self).__init__()
        
        self.fc1 = nn.Linear(n_in[1], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_out)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):

        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)
        #out = F.log_softmax(out)
        #out = F.log_softmax(out, dim=1)
        return out

