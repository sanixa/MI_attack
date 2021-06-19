import sys
from math import ceil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, TensorDataset
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
#from pyvacy.pyvacy import optim, analysis, sampling
import math
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"]="5"




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
        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

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

    def forward(self, x):

        x = self.model(x)

        return x


def grad_dp_lay(model, inputs, labels, loss_function, num_classes, gradient_norm_clip=1., beta=1.):        
    outputs = model(inputs)
    #print(inputs.shape, outputs.shape, labels.shape)
    #calculate per example gradient
    per_grad_list=[]
    total_loss = 0
    for idx in range(len(inputs)):
        model.zero_grad()
        per_outputs=outputs[idx].view(-1,num_classes)
        per_labels=labels[idx].view(-1)
        per_loss = loss_function(per_outputs, per_labels)
        total_loss += per_loss
        per_loss.backward(retain_graph=True)

        ###gradient encoding
        g_vec = torch.cat([p.grad.view(-1) for p in model.parameters()])
        dist_mat = model.cdist(g_vec.unsqueeze(0), model.grads)
        #import ipdb; ipdb.set_trace()
        closest_idx = int(dist_mat.argmax().data.cpu().numpy())

        for idx, p in enumerate(model.parameters()):
            p.grad = model.grad_dict[closest_idx][idx]

        per_grad=[]
        iter_num=0
        per_grad_norm=0           
        for param in model.parameters():
            per_grad.append(copy.deepcopy(param.grad.data))
            per_grad_norm = per_grad[iter_num].norm(2)
            clip_val=max(gradient_norm_clip,per_grad_norm/beta)
            per_grad[iter_num]=per_grad[iter_num]/clip_val
            iter_num+=1
        per_grad_list.append(per_grad)
    batch_grad=[]
    for param_num in range(len(per_grad_list[0])):
        temp_tenor=per_grad_list[0][param_num]
        for idx in range(len(per_grad_list)-1):
            temp_tenor=temp_tenor+per_grad_list[idx+1][param_num]
        batch_grad.append(temp_tenor)        
    return batch_grad, per_grad_norm, total_loss

def train(args, model, train_loader, opt, epoch, max_sensitivity):
    model.train()

    criterion = nn.CrossEntropyLoss()

    beta = 1.

    for batch_idx, (data, label) in enumerate(train_loader):
        opt.zero_grad()
        data, label = data.cuda(), label.cuda()

        batch_grad, _, loss=grad_dp_lay(model,data,label,criterion,10,args.clip,beta)

        iter_grad=0
        for p in model.parameters():
            size_param = torch.numel(p)
            tmp = p.grad.view(-1, size_param)
            noise_g = beta*args.noise*max_sensitivity*torch.zeros_like(tmp.data).normal_()
            p.grad.data = (batch_grad[iter_grad].view(-1, size_param)/args.minibatch+noise_g/args.minibatch).view(p.grad.size())
            iter_grad=iter_grad+1
    
        loss_value = loss / args.minibatch

        opt.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{10000} ({batch_idx * len(data) / 10000.:.0f})]\tLoss: {loss.item():.7f}')


def collect(args, model, train_loader, opt, epoch, grad_dict_num):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(), label.cuda()
        output = model(data)
        loss = criterion(output, label)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if torch.rand(1) < 0.2:
            model.grad_dict[grad_dict_num] = [p.grad.clone() for p in model.parameters()]
            g_vec = torch.cat([g.view(-1) for g in model.grad_dict[grad_dict_num]])
            model.grads.append(g_vec)
            grad_dict_num += 1

        if batch_idx % args.log_interval == 0:
            print(f'collect epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]')
    return grad_dict_num


def test(model, test_loader):
    global fl
    model.eval()
    correct = 0.

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()


    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(correct), len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    fl.write("Test set: Accuracy: {}/{} ({:.2f}%)\n".format(
        int(correct), len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

def main():
    global fl
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dir', type=str, default='result')
    parser.add_argument('--minibatch', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 60')
    parser.add_argument('--microbatch', type=int, default=1, metavar='N',
                        help='input micro size for training (default: 1')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument("--noise", type=float, default=1.1)
    parser.add_argument('--clip', type=float, default=4)
    parser.add_argument('--l2-penalty', type=float, default=0.001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    train_kwargs = {'batch_size': args.minibatch}
    test_kwargs = {'batch_size': args.test_batch_size}
    noise = args.noise

    eps = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n = 60000, 
                                                  batch_size = args.minibatch, 
                                                  noise_multiplier = args.noise, 
                                                  epochs = args.epochs, 
                                                  delta = 1e-5)

    print(f'noise: {args.noise}\neps: {eps[0]}\n')
    #print(aaa)
    fl = open(f'{args.dir}/cifar10_noise{args.noise}_eps{eps[0]:.2f}_epoch{args.epochs}.txt', 'w')

    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        ])


    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10('./data', train=False, transform=transform)
    '''
    TargetTrainData = train_data_array[40000:50000]
    TargetTrainLabel = train_targets_array[40000:50000]
    TargetTestData = test_data_array
    TargetTestLabel = test_targets_array
    ShadowTrainData = train_data_array[:10000]
    ShadowTrainLabel = train_targets_array[:10000]
    ShadowTestData = train_data_array[10000:20000]
    ShadowTestLabel = train_targets_array[10000:20000]
    '''
    test_dataset.data = train_dataset.data[10000:20000]
    test_dataset.targets = train_dataset.targets[10000:20000]
    train_dataset.data = train_dataset.data[:10000]
    train_dataset.targets = train_dataset.targets[:10000]


    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = ResNet18v2_cifar10(0).cuda()
    opt = optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.l2_penalty,
    )
    grad_dict_num = 0
    for i in range(30):
        grad_dict_num = collect(args, model, train_loader, opt, i, grad_dict_num)
    print("finish collecting")
    model.grads = torch.stack(model.grads)

    sensitivity = []
    print(len(model.grads))
    for i in range(len(model.grads)):
        temp = 0.
        for j in range(len(model.grad_dict[i])):
            temp += torch.norm(model.grad_dict[i][j], p=2)**2
        sensitivity.append(temp**0.5)

    print(max(sensitivity))
    print("test")
    print(min(sensitivity))
    print("test2")

    opt = optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.l2_penalty,
    )
    scheduler = StepLR(opt, step_size=5, gamma=args.gamma)
    
    bool_first_exceed = [0,0,0,0,0,0,0,0,0] #0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, opt, epoch, max(sensitivity))
        acc = test(model, test_loader)
        scheduler.step()

        acc = int(acc * 10)
        if bool_first_exceed[acc-1] != 1:
            bool_first_exceed[acc-1] = 1
            torch.save(model.state_dict(), args.dir + "/cifar10_model_acc_0." + str(acc) + ".pt")
        if epoch in [10,50,100,200]:
            torch.save(model.state_dict(), args.dir + "/cifar10_model_epoch_" + str(epoch) + ".pt")

    if args.save_model:
        torch.save(model.state_dict(), args.dir + "/cifar10_model.pt")
    fl.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise
        fl.close()

