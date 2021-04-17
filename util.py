import logging
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import copy,os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,5'

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class MyCustomDataset(Dataset):
    def __init__(self, data_source, transforms=None):
        # stuff
        self.X, self.Y = data_source
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        label = self.Y[index]
        data = self.X[index] # Some data read from a file or image
        if self.transforms is not None:
            data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        img = data
        return (img, label)

    def __len__(self):
        count = len(self.X)
        return count # of how many data(images?) you have

def construct_dataset(dataset, data_source=None, data_transform=1):
    transform = None
    if dataset == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
    elif dataset == 'cifar_100':
        transform_=transforms.Compose([
            #transforms.Resize((160, 160)),
            #transforms.RandomCrop((128, 128)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
    elif dataset == 'cifar_10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform=transforms.Compose([])

    num_classes = 0
    if data_source is None:
        if dataset == 'mnist':
            train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transform_train)
            test_dataset = datasets.MNIST('../data', train=False,
                        transform=transform_test)
            num_classes = 10
        elif dataset == 'cifar_100':
            train_dataset = datasets.CIFAR100('../data', train=True, download=True,
                        transform=transform_train)
            test_dataset = datasets.CIFAR100('../data', train=False,
                        transform=transform_test)
            num_classes = 100
        elif dataset == 'cifar_10':
            train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform_train)
            test_dataset = datasets.CIFAR10('../data', train=False,
                        transform=transform_test)
            num_classes = 10
    else:
        Train, TrainLabel, Test, TestLabel = data_source
        unique_classes = np.unique(TrainLabel)
        num_classes = len(unique_classes)
        train_dataset = MyCustomDataset((Train.copy(), TrainLabel.copy()), None)
        test_dataset = MyCustomDataset((Test.copy(), TestLabel.copy()), None)
    return train_dataset, test_dataset, num_classes



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

GRADIENT_NORM_CLIP = 1.
SENSITIVITY = 2.

def train(model, device, train_loader, optimizer, epoch, model_param_prune, gradient_prune, batch_size, nm, loss_function, logger, num_classes, log_interval=10):
    model.train()
    
    # import ipdb; ipdb.set_trace()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.long().to(device)
        #y_onehot = torch.zeros([batch_size, 10]).to(device)
        #target = y_onehot.scatter_(1, target.view(-1,1), 1)
        optimizer.zero_grad()
        loss_value = 0

        if model_param_prune:
            #clip by threshold, model weight
            with torch.no_grad():
                for idx, p in enumerate(model.parameters()):
                    shape = p.shape
                    temp = p.view(-1)
                    thres = np.mean(temp.cpu().numpy())# + np.std(temp.cpu().numpy())
                    #thres = 1e-4
                    for i in range(len(temp)):
                        if temp[i] < thres:
                            temp[i] = 0
                    p = temp.view(shape)
        
        if nm > 0:
            beta = 1
            iter_grad=0
            batch_grad, _, loss=grad_dp_lay(model,data,target,loss_function,num_classes,GRADIENT_NORM_CLIP,beta)
            loss_value += loss

            for p in model.parameters():
                size_param = torch.numel(p)
                tmp = p.grad.view(-1, size_param)
                noise_g = beta*nm*SENSITIVITY*torch.zeros_like(tmp.data).normal_()
                p.grad.data = (batch_grad[iter_grad].view(-1, size_param)/batch_size+noise_g/batch_size).view(p.grad.size())
                iter_grad=iter_grad+1

                if gradient_prune:
                    #sort and clip
                    shape = p.grad.shape
                    temp = p.grad.view(-1)
                    sign = torch.ones_like(temp)
                    ##store sign
                    for i in range(len(temp)):
                        if temp[i] < 0:
                            sign[i] = -1
                
                    temp = torch.abs(temp)
                    idx = torch.argsort(temp, dim=0)
                    for i in range(len(idx)):
                        if idx[i] < len(temp) * 0.5:
                            temp[i] = 0
                        else:
                            temp[i] = temp[i] * sign[i]
                    p.grad = temp.view(shape)

        else:
            output = model(data)
            
            ##BCE loss needs specific label
            if type(loss_function) is torch.nn.modules.loss.BCELoss or type(loss_function) is torch.nn.modules.loss.BCEWithLogitsLoss:
                #target_onehot = torch.zeros([target.shape[0], 2]).to(device)
                #target = target_onehot.scatter_(1, target.view(-1,1), 1)
                target = target.view(-1,1).float()
            
            loss = loss_function(output, target)
            
            loss.backward()
            if gradient_prune:
                for idx, p in enumerate(model.parameters()):
                    shape = p.grad.shape

                    #sort and clip
                    shape = p.grad.shape
                    temp = p.grad.view(-1)
                    sign = torch.ones_like(temp)
                    ##store sign
                    for i in range(len(temp)):
                        if temp[i] < 0:
                            sign[i] = -1
                
                    temp = torch.abs(temp)
                    idx = torch.argsort(temp, dim=0)
                    for i in range(len(idx)):
                        if idx[i] < len(temp) * 0.5:
                            temp[i] = 0
                        else:
                            temp[i] = temp[i] * sign[i]
                    p.grad = temp.view(shape)

        loss_value = loss.item() / batch_size
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_value))
    ### cal train acc
    correct = 0
    train_loss = 0

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)
            
            if type(loss_function) is torch.nn.modules.loss.BCELoss or type(loss_function) is torch.nn.modules.loss.BCEWithLogitsLoss:
                ##BCE loss needs specific label
                #target_onehot = torch.zeros([target.shape[0], 2]).to(device)
                #target_onehot = target_onehot.scatter_(1, target.view(-1,1), 1)
                target = target.view(-1,1).float()
                train_loss += loss_function(output, target)
                assert target.size() == output.size()
                pred = output > 0.5
            else:
            
                train_loss += loss_function(output, target) 
                pred = output.argmax(dim=1, keepdim=True)  


            correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)

    logger.info('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return 100. * correct / len(train_loader.dataset)


def test(model, device, test_loader, batch_size, loss_function, logger):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)

            
            if type(loss_function) is torch.nn.modules.loss.BCELoss or type(loss_function) is torch.nn.modules.loss.BCEWithLogitsLoss:
                ##BCE loss needs epecific label
                #target_onehot = torch.zeros([target.shape[0], 2]).to(device)
                #target_onehot = target_onehot.scatter_(1, target.view(-1,1), 1)
                target = target.view(-1,1).float()
                test_loss += loss_function(output, target) 
                assert target.size() == output.size()
                pred = output > 0.5
            else:   
                test_loss += loss_function(output, target) 
                pred = output.argmax(dim=1, keepdim=True)  

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)