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


def Laplacian_smoothing(net, sigma=1):
    ## after add dp noise
    for p_net in net.parameters():
        size_param = torch.numel(p_net)
        tmp = p_net.grad.view(-1, size_param)

        c = np.zeros(shape=(1, size_param))
        c[0, 0] = -2.; c[0, 1] = 1.; c[0, -1] = 1.
        c = torch.Tensor(c).cuda()
        zero_N = torch.zeros(1, size_param).cuda()
        c_fft = torch.rfft(c, 1, onesided=False)
        coeff = 1./(1.-sigma*c_fft[...,0])
        ft_tmp = torch.rfft(tmp, 1, onesided=False)
        tmp = torch.zeros_like(ft_tmp)
        tmp[...,0] = ft_tmp[...,0]*coeff
        tmp[...,1] = ft_tmp[...,1]*coeff
        tmp = torch.irfft(tmp, 1, onesided=False)
        tmp = tmp.view(p_net.grad.size())
        p_net.grad.data = tmp

    return net

def model_pruning(net, clip_rate=0.5):
    with torch.no_grad():
        for p_net in net.parameters():
            shape = p_net.grad.shape
            temp = p_net.grad.view(-1)
            sign = torch.ones_like(temp)
            ##store sign
            for i in range(len(temp)):
                if temp[i] < 0:
                    sign[i] = -1
        
            temp = torch.abs(temp)
            idx = torch.argsort(temp, dim=0)
            for i in range(len(idx)):
                if idx[i] < len(temp) * clip_rate:
                    temp[i] = 0
                else:
                    temp[i] = temp[i] * sign[i]
            p_net = temp.view(shape)
    return net

##considering computing time, not sort every batch but clip by a threshold
def model_pruning_relax(net, epoch, thres=None):
    if epoch < 10:  ## by algo. 2.
        return net
    else:
        with torch.no_grad():
            for p_net in net.parameters():
                #import ipdb; ipdb.set_trace()
                shape = p_net.shape
                temp = p_net.view(-1)
                if thres is None:
                    #thres = np.mean(temp.cpu().numpy())# + np.std(temp.cpu().numpy())
                    thres = 1e-4
                for i in range(len(temp)):
                    if torch.abs(temp[i]) < thres:
                        temp[i] = 0
                p_net = temp.view(shape)
        return net

def Gradient_encoding_collect(model, device, train_loader, optimizer, scheduler, loss_function, epochs, logger):
    model.train()
    # import ipdb; ipdb.set_trace()
    SAMPLE_NUM = 1000
    grad_dict = {}
    grads = []
    iter_num=0
    max_grad_norm_list = []

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            #import ipdb; ipdb.set_trace()
            grad_dict[iter_num] = [p.grad.clone() for p in model.parameters()]
            max_grad_norm_list.append([np.linalg.norm(p.grad.clone().cpu().numpy().reshape(-1), 2) for p in model.parameters()])
            g_vec = torch.cat([g.view(-1) for g in grad_dict[batch_idx]])
            grads.append(g_vec)
            iter_num += 1
            optimizer.step()
            scheduler.step()
            if batch_idx % 10 == 0:
                print('Collect Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()/ len(data)))

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


    #import ipdb; ipdb.set_trace()
    shuffle_indices = np.arange(iter_num)
    np.random.shuffle(shuffle_indices)
    idx_list = np.array(range(iter_num))[shuffle_indices][:SAMPLE_NUM]
    ## sample the number SAMPLE_NUM of grads
    grads = [grads[i] for i in range(len(grads)) if i in idx_list]
    max_grad_norm_list = [max_grad_norm_list[i] for i in range(len(max_grad_norm_list)) if i in idx_list]
    ## calculate max grad l2 norm
    max_grad_norm = np.max(np.array(max_grad_norm_list).reshape(-1))
    
    logger.info('\nmax_grad_norm ({:.4f})\n'.format(max_grad_norm))

    return grad_dict, grads, max_grad_norm, idx_list

def Gradient_encoding_apply(model, grad_dict, grads, idx_list, device):
    g_vec = torch.cat([p.grad.view(-1) for p in model.parameters()])
    cos = torch.nn.CosineSimilarity()
    dist_mat = cos(g_vec.unsqueeze(0), torch.tensor([item.cpu().detach().numpy() for item in grads]).to(device))
    #import ipdb; ipdb.set_trace()
    min_idx = int(dist_mat.argmin().data.cpu().numpy())
    #bk_vec = grads[min_idx,:] # Debug
    for idx, p in enumerate(model.parameters()):
        p.grad = grad_dict[idx_list[min_idx]][idx]

    return model



def train(sub_method, model, device, train_loader, optimizer, scheduler, epoch, batch_size, nm, loss_function, logger, num_classes, log_interval=10, grad_dict=None, grads=None, max_grad_norm=None, idx_list=None):
    model.train()
    GRADIENT_NORM_CLIP = 1.
    SENSITIVITY = 2.

    # import ipdb; ipdb.set_trace()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.long().to(device)
        #y_onehot = torch.zeros([batch_size, 10]).to(device)
        #target = y_onehot.scatter_(1, target.view(-1,1), 1)
        optimizer.zero_grad()
        loss_value = 0

        
        if nm > 0:
            beta = 1
            ### Gradient Encoding
            if sub_method[2]:

                output = model(data)
                ##BCE loss needs specific label
                if type(loss_function) is torch.nn.modules.loss.BCELoss or type(loss_function) is torch.nn.modules.loss.BCEWithLogitsLoss:
                    #target_onehot = torch.zeros([target.shape[0], 2]).to(device)
                    #target = target_onehot.scatter_(1, target.view(-1,1), 1)
                    target = target.view(-1,1).float()
                loss = loss_function(output, target)
                loss.backward()
                loss_value = loss.item() / batch_size
                import ipdb; ipdb.set_trace()
                model = Gradient_encoding_apply(model, grad_dict, grads, idx_list, device)
                SENSITIVITY = max_grad_norm * 2

                clip_bound_ = GRADIENT_NORM_CLIP / batch_size
                SENSITIVITY = GRADIENT_NORM_CLIP * 2

                for p in model.parameters():
                    size_param = torch.numel(p)
                    tmp = p.grad.view(-1, size_param)
                    ### clip
                    grad_norm = torch.norm(tmp.data, p=2, dim=1)

                    clip_coef = clip_bound_ / (grad_norm + 1e-10)
                    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
                    clip_coef = clip_coef.unsqueeze(-1)
                    tmp.data = clip_coef * tmp.data
                    ### noise
                    noise_g = beta*nm*SENSITIVITY*torch.zeros_like(tmp.data).normal_()
                    p.grad.data = (tmp.data/batch_size+noise_g/batch_size).view(p.grad.size())
            else:
                batch_grad, _, loss=grad_dp_lay(model,data,target,loss_function,num_classes,GRADIENT_NORM_CLIP,beta)

                iter_grad=0
                for p in model.parameters():
                    size_param = torch.numel(p)
                    tmp = p.grad.view(-1, size_param)
                    noise_g = beta*nm*SENSITIVITY*torch.zeros_like(tmp.data).normal_()
                    p.grad.data = (batch_grad[iter_grad].view(-1, size_param)/batch_size+noise_g/batch_size).view(p.grad.size())
                    iter_grad=iter_grad+1
            
                loss_value += loss
            ### Laplacian_smoothing
            if sub_method[0]: 
                model = Laplacian_smoothing(model)

        else:
            output = model(data)
            
            ##BCE loss needs specific label
            if type(loss_function) is torch.nn.modules.loss.BCELoss or type(loss_function) is torch.nn.modules.loss.BCEWithLogitsLoss:
                #target_onehot = torch.zeros([target.shape[0], 2]).to(device)
                #target = target_onehot.scatter_(1, target.view(-1,1), 1)
                target = target.view(-1,1).float()
            
            loss = loss_function(output, target)
            loss.backward()

            loss_value = loss.item() / batch_size
        optimizer.step()
        scheduler.step()


        ##model parameter pruning
        if sub_method[3]:
        #clip model weight by threshold, speed up compared to clip by rate
            model = model_pruning_relax(model, epoch)

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