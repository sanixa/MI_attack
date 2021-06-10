from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
import scipy.stats as stats
import os, random, time
from torchsummary import summary
from torch.utils.data.dataset import Dataset

from torch.utils.data.sampler import SubsetRandomSampler


from rdp_accountant_nn import compute_rdp
from rdp_accountant_nn import get_privacy_spent

import attack,util
from model import *
from torch.optim import *
from torch.nn import *
import torchvision.models as models
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,5'



def train(train_param):

    (sub_method, dataset, dataset_custom, data_source, data_transform, id, model_name, model_type, summary_model_shape, seed, nm, ep, batch_size, test_batch_size, epochs, param, optimizer_name, lr, loss_function_name, gamma, log_interval, logger, save_model) = train_param

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def compute_epsilon(steps,nm):
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        sampling_probability = batch_size/60000
        rdp = compute_rdp(q=sampling_probability,
                        noise_multiplier=nm,
                        steps=steps,
                        orders=orders)
        #Delta is set to 1e-5 because MNIST has 60000 training points.
        return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

    ### compute stop cond first
    if nm != -1: 
        eps = compute_epsilon((epochs+1) * 60000 / batch_size,nm)
        logger.info('epsilon: ' + str(eps))
        print(eps)
        #print(aaa)

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    ## dataset
    num_classes = 0
    if not dataset_custom:
        train_dataset, test_dataset, num_classes = util.construct_dataset(dataset)
    else:
        train_dataset, test_dataset, num_classes = util.construct_dataset(dataset, data_source, data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    if model_type == 'softmax' and summary_model_shape == None:
        temp = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        data, target = next(iter(temp))
        summary_model_shape = tuple(data.shape)

        model =  globals()[model_name](data.shape, 1).to(device)
        summary(model, summary_model_shape)
    else:
        if sub_method[1]: ##Tempered Sigmoid
            model_name = model_name + '_TS'
            model =  globals()[model_name](param).to(device)
            summary(model, summary_model_shape)
        else:
            model =  globals()[model_name](param).to(device)
            summary(model, summary_model_shape)

    optimizer = globals()[optimizer_name](model.parameters(), lr)
    loss_function = globals()[loss_function_name](reduction='sum')
    scheduler = StepLR(optimizer, step_size=50, gamma=0.7)

    #train phase
    
    model_path = "model/" + str(id) + "_dataset_" + dataset + "_ep_" + str(ep) + "_nm_" + str(nm) + \
        "_epoch_" + str(epochs) + "_" + str(time.time()) + "/"
    sub_method_name = ['Laplacian_Smoothing', 'Tempered_Sigmoid' , 'Gradient_Encoding', 'model_pruning']
    for i in range(len(sub_method)):
        if sub_method[i]:
            model_path = model_path[:-1] + "_" + sub_method_name[i] + "/"
    os.makedirs(model_path, exist_ok=True) 
    # slide window for calculate past average test acc, if exceed specific threshold then save model
    slide_window = 1
    acc_test_list = []
    bool_first_exceed = [0,0,0,0,0,0,0,0,0] #0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9

    ### Gradient Encoding, collect grad before dp training
    if sub_method[2]:
        epochs_GEC = 50
        model_GEC = copy.deepcopy(model)
        optimizer_GEC = globals()[optimizer_name](model_GEC.parameters(), lr)
        scheduler_GEC = StepLR(optimizer_GEC, step_size=100, gamma=0.7)
        grad_dict, grads, max_grad_norm, idx_list = util.Gradient_encoding_collect(model_GEC, device, train_loader, optimizer_GEC, scheduler_GEC, loss_function, epochs_GEC, logger)
    
    #import ipdb; ipdb.set_trace()
    for epoch in range(1, epochs + 1):
        if sub_method[2]:
            acc_train = util.train(sub_method, model, device, train_loader, optimizer, scheduler, epoch, batch_size, nm, loss_function, logger, num_classes, log_interval, grad_dict, grads, max_grad_norm, idx_list)
        else:
            acc_train = util.train(sub_method, model, device, train_loader, optimizer, scheduler, epoch, batch_size, nm, loss_function, logger, num_classes, log_interval)
        acc_test = util.test(model, device, test_loader, test_batch_size, loss_function, logger)
        acc_test_list.append(acc_test)
        acc = int(np.mean(acc_test_list[-slide_window:]) * 10)
        if bool_first_exceed[acc-1] != 1:
            bool_first_exceed[acc-1] = 1
            torch.save(model.state_dict(), model_path + "model_acc_0." + str(acc) + ".pt")
            logger.info('save model/acc: ' + str(acc_test) + '/epoch: ' + str(epoch))
        if epoch in [10,50,100,200]:
            torch.save(model.state_dict(), model_path + "model_epoch_" + str(epoch) + ".pt")
            logger.info('save model/epoch: ' + str(epoch))

    ##model parameter pruning
    if sub_method[3]:
    #clip model weight by rate to fix weight sparsity
        model = util.model_pruning(model)
    
    new_model_path = model_path[:-1] + '_' + str(acc_train) + '_' + str(acc_test)+ '/'
    os.rename(model_path, new_model_path) 
    if save_model:
        torch.save(model.state_dict(),  new_model_path + "model.pt")
    
    return model, new_model_path

