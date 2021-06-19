import sys
from math import ceil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, TensorDataset
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from pyvacy.pyvacy import optim, analysis, sampling
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Flatten(nn.Module):
    def forward(self, data):
        return data.reshape(data.shape[0], -1)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 8, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            Flatten(),
            nn.Linear(288, 10),
            nn.LogSoftmax(dim=1)
        )

        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

    def forward(self, data):
        return self.model(data)

class ConvNet_mnist(nn.Module):
    
    def __init__(self, param, num_classes=10):

        super(ConvNet_mnist, self).__init__()
        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

        
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        #out = F.log_softmax(out, dim=1)
        return out



def train(args, model, train_dataset, opt, epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()

    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        args.minibatch,
        args.microbatch,
        args.epochs
    )

    for mini_idx, (x_mini, y_mini) in enumerate(minibatch_loader(train_dataset)):
        opt.zero_grad()
        for x_micro, y_micro in microbatch_loader(TensorDataset(x_mini, y_mini)):
            x_micro, y_micro = x_micro.cuda(), y_micro.cuda()
            opt.zero_microbatch_grad()
            loss = criterion(model(x_micro), y_micro)
            loss.backward()

            g_vec = torch.cat([p.grad.view(-1) for p in model.parameters()])
            dist_mat = model.cdist(g_vec.unsqueeze(0), model.grads)
            #import ipdb; ipdb.set_trace()
            closest_idx = int(dist_mat.argmax().data.cpu().numpy())

            for idx, p in enumerate(model.parameters()):
                 p.grad = model.grad_dict[closest_idx][idx]

            opt.microbatch_step()
        opt.step()

        if mini_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{mini_idx * len(x_mini)}/{10000} ({mini_idx * len(x_mini) / 10000.:.0f})]\tLoss: {loss.item():.7f}')


def collect(args, model, train_loader, opt, epoch):
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i in range(10):
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = criterion(output, label)

            opt.zero_microbatch_grad()
            loss.backward()
            opt.step()
            model.grad_dict[batch_idx+i*ceil(10000/args.minibatch)] = [p.grad.clone() for p in model.parameters()]
            g_vec = torch.cat([g.view(-1) for g in model.grad_dict[batch_idx+i*ceil(10000/args.minibatch)]])
            model.grads.append(g_vec)

            if batch_idx % args.log_interval == 0:
                print(f'collect epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]')



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
    fl = open(f'{args.dir}/mnist_noise{args.noise}_eps{eps[0]:.2f}_epoch{args.epochs}.txt', 'w')

    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        #transforms.Normalize((0.5,), (0.5,))
        ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
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

    model = ConvNet_mnist(0).cuda()
    opt = optim.DPSGD(
        l2_norm_clip=args.clip,
        noise_multiplier=args.noise,
        minibatch_size=args.minibatch,
        microbatch_size=args.microbatch,
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.l2_penalty,
    )

    collect(args, model, train_loader, opt, epoch=10)
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

    opt = optim.DPSGD(
        l2_norm_clip=max(sensitivity),
        noise_multiplier=args.noise,
        minibatch_size=args.minibatch,
        microbatch_size=args.microbatch,
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.l2_penalty,
    )
    scheduler = StepLR(opt, step_size=5, gamma=args.gamma)
    
    bool_first_exceed = [0,0,0,0,0,0,0,0,0] #0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_dataset, opt, epoch)
        acc = test(model, test_loader)
        scheduler.step()

        acc = int(acc * 10)
        if bool_first_exceed[acc-1] != 1:
            bool_first_exceed[acc-1] = 1
            torch.save(model.state_dict(), args.dir + "/model_acc_0." + str(acc) + ".pt")
        if epoch in [10,50,100,200]:
            torch.save(model.state_dict(), args.dir + "/model_epoch_" + str(epoch) + ".pt")

    if args.save_model:
        torch.save(model.state_dict(), args.dir + "/model.pt")
    fl.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise
        fl.close()


