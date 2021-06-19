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
from pyvacy.pyvacy import optim, analysis, sampling

os.environ['CUDA_VISIBLE_DEVICES'] = '5'



def train(args, sub_method, model_id, model_name, model_type, data_source, logger):

    #(sub_method, dataset, dataset_custom, data_source, data_transform, id, model_name, model_type, summary_model_shape, seed, nm, ep, batch_size, test_batch_size, epochs, param, optimizer_name, lr, loss_function_name, gamma, log_interval, logger, save_model) = train_param

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def compute_epsilon(steps,nm):
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        sampling_probability = args.batch_size/60000
        rdp = compute_rdp(q=sampling_probability,
                        noise_multiplier=nm,
                        steps=steps,
                        orders=orders)
        #Delta is set to 1e-5 because MNIST has 60000 training points.
        return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

    ### compute stop cond first
    if args.nm != -1: 
        eps = compute_epsilon((args.epochs+1) * 60000 / args.batch_size,args.nm)
        logger.info('epsilon: ' + str(eps))
        print(eps)
        #print(aaa)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    ## dataset
    num_classes = 0
    train_dataset, test_dataset, num_classes = util.construct_dataset(args.dataset, data_source)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    if model_type == 'binary_class':
        temp = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        data, target = next(iter(temp))
        summary_model_shape = tuple(data.shape)

        model =  globals()[model_name](data.shape, 1).to(device)
        summary(model, summary_model_shape)
    else:
        if args.dataset == 'mnist':
            summary_model_shape = (1,28,28)
        elif args.dataset == 'cifar_10' or args.dataset == 'cifar_100':
            summary_model_shape = (3,32,32)
        if sub_method[1]: ##Tempered Sigmoid
            model_name = model_name + '_TS'
            model =  globals()[model_name](args.param).to(device)
            summary(model, summary_model_shape)
        else:
            model =  globals()[model_name](args.param).to(device)
            summary(model, summary_model_shape)
    if args.dataset == 'cifar_10':
        optimizer = globals()['Adam'](model.parameters(), args.lr)
    elif  args.dataset == 'mnist':
        optimizer = optim.DPSGD(
            l2_norm_clip=1,
            noise_multiplier=args.nm,
            minibatch_size=args.batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=args.lr,
            weight_decay=0.001,
        )
    loss_function = globals()[args.loss_name](reduction='sum')
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

    #train phase
    
    model_path = "model/" + str(model_id) + "_dataset_" + args.dataset + "_ep_" + str(args.ep) + "_nm_" + str(args.nm) + \
        "_epoch_" + str(args.epochs) + "_" + str(time.time()) + "/"
    sub_method_name = ['Laplacian_Smoothing', 'Tempered_Sigmoid' , 'Gradient_Encoding', 'model_pruning']
    for i in range(len(sub_method)):
        if sub_method[i]:
            model_path = model_path[:-1] + "_" + sub_method_name[i] + "/"
    os.makedirs(model_path, exist_ok=True) 


    ### Gradient Encoding, collect grad before dp training
    if sub_method[2]:
        epochs_GEC = 15
        grad_dict, grads, max_grad_norm, idx_list = util.Gradient_encoding_collect(model, device, train_loader, args.batch_size, optimizer, scheduler, loss_function, epochs_GEC, logger)
    
    # slide window for calculate past average test acc, if exceed specific threshold then save model
    slide_window = 1
    acc_test_list = []
    bool_first_exceed = [0,0,0,0,0,0,0,0,0] #0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9

    #import ipdb; ipdb.set_trace()
    for epoch in range(1, args.epochs + 1):
        if sub_method[2]:
            acc_train = util.train(args, sub_method, model, device, train_dataset, train_loader, optimizer, loss_function, epoch, num_classes, logger, grad_dict, grads, max_grad_norm, idx_list)
        else:
            acc_train = util.train(args, sub_method, model, device, train_dataset, train_loader, optimizer, loss_function, epoch, num_classes, logger)
        acc_test = util.test(args, model, device, test_loader, loss_function, logger)
        acc_test_list.append(acc_test)
        acc = int(np.mean(acc_test_list[-slide_window:]) * 10)
        if bool_first_exceed[acc-1] != 1:
            bool_first_exceed[acc-1] = 1
            torch.save(model.state_dict(), model_path + "model_acc_0." + str(acc) + ".pt")
            logger.info('save model/acc: ' + str(acc_test) + '/epoch: ' + str(epoch))
        if epoch in [10,50,100,200]:
            torch.save(model.state_dict(), model_path + "model_epoch_" + str(epoch) + ".pt")
            logger.info('save model/epoch: ' + str(epoch))
        scheduler.step()
    ##model parameter pruning
    if sub_method[3]:
    #clip model weight by rate to fix weight sparsity
        model = util.model_pruning(model)
    
    new_model_path = model_path[:-1] + '_' + str(acc_train) + '_' + str(acc_test)+ '/'
    os.rename(model_path, new_model_path) 
    if args.save_model:
        torch.save(model.state_dict(),  new_model_path + "model.pt")
    
    return model, new_model_path

