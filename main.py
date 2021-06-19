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
from sklearn.preprocessing import minmax_scale,MaxAbsScaler
import scipy.stats as stats
import os, random, time
from torchsummary import summary


from rdp_accountant_nn import compute_rdp
from rdp_accountant_nn import get_privacy_spent

import attack,trainNN,util
import model

import matplotlib.pyplot as plt
from sklearn import manifold
import torchvision.models as models
from shutil import copyfile


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
dataPath = './data/'


def clipDataTopX(dataToClip, top=3):
    #sorted(s, key=lambda x:(int(x<0), abs(x)))
    #
    res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
    return np.array(res)

def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y

def initializeData(dataset, logger):
    if dataset == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('../data', train=False,
                        transform=transform)

    elif dataset == 'cifar_100':
        transform_train=transforms.Compose([
            #transforms.Resize((160, 160)),
            #transforms.RandomCrop((128, 128)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        transform_test=transforms.Compose([
            #transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ])
        train_dataset = datasets.CIFAR100('../data', train=True, download=True,
                        transform=transform_train)
        test_dataset = datasets.CIFAR100('../data', train=False,
                        transform=transform_test)

    elif dataset == 'cifar_10':
        transform_train=transforms.Compose([
            #transforms.Resize((160, 160)),
            #transforms.RandomCrop((128, 128)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        transform_test=transforms.Compose([
            #transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ])
        train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform_train)
        test_dataset = datasets.CIFAR10('../data', train=False,
                        transform=transform_test)

    train_data_array = train_dataset.data
    test_data_array = test_dataset.data
    if dataset != 'mnist':
        train_data_array = train_data_array.transpose((0,3,1,2))
        test_data_array = test_data_array.transpose((0,3,1,2))
    else:
        train_data_array = train_data_array.view((-1,1,28,28))
        test_data_array = test_data_array.view((-1,1,28,28))

    train_targets_array = np.array(train_dataset.targets)
    test_targets_array = np.array(test_dataset.targets)



    ##split target and shadow data
    #shuffle
    train_indices = np.arange(train_data_array.shape[0])
    np.random.shuffle(train_indices)
    train_data_array = train_data_array[train_indices]
    train_targets_array = train_targets_array[train_indices]

    test_indices = np.arange(test_data_array.shape[0])
    np.random.shuffle(test_indices)
    test_data_array = test_data_array[test_indices]
    test_targets_array = test_targets_array[test_indices]


    #split
    '''
    #### each class split to target/shadow equally (train)
    unique_classes = np.unique(train_targets_array)
    indices = np.arange(len(train_targets_array))
    TargetTrainData, TargetTrainLabel, ShadowTrainData,ShadowTrainLabel = None, None, None, None

    for c in unique_classes:
        c_indices = indices[train_targets_array == c]
        c_x, c_y = train_data_array[c_indices], train_targets_array[c_indices]
        
        split_idx = int(c_x.shape[0] * 0.5)
        TargetData_temp = np.array(c_x[:split_idx])
        TargetLabel_temp = np.array(c_y[:split_idx])
        ShadowData_temp = np.array(c_x[split_idx:])
        ShadowLabel_temp = np.array(c_y[split_idx:])
        if TargetTrainData is None:
            TargetTrainData = TargetData_temp
            TargetTrainLabel = TargetLabel_temp
            ShadowTrainData = ShadowData_temp
            ShadowTrainLabel = ShadowLabel_temp
        else:
            TargetTrainData = np.concatenate((TargetTrainData, TargetData_temp))
            TargetTrainLabel = np.concatenate((TargetTrainLabel, TargetLabel_temp))
            ShadowTrainData = np.concatenate((ShadowTrainData, ShadowData_temp))
            ShadowTrainLabel = np.concatenate((ShadowTrainLabel, ShadowLabel_temp))

    #### each class split to target/shadow equally (test)
    unique_classes = np.unique(test_targets_array)
    indices = np.arange(len(test_targets_array))
    TargetTestData, TargetTestLabel, ShadowTestData,ShadowTestLabel = None, None, None, None

    for c in unique_classes:
        c_indices = indices[test_targets_array == c]
        c_x, c_y = test_data_array[c_indices], test_targets_array[c_indices]
        
        split_idx = int(c_x.shape[0] * 0.5)
        TargetData_temp = np.array(c_x[:split_idx])
        TargetLabel_temp = np.array(c_y[:split_idx])
        ShadowData_temp = np.array(c_x[split_idx:])
        ShadowLabel_temp = np.array(c_y[split_idx:])
        if TargetTestData is None:
            TargetTestData = TargetData_temp
            TargetTestLabel = TargetLabel_temp
            ShadowTestData = ShadowData_temp
            ShadowTestLabel = ShadowLabel_temp
        else:
            TargetTestData = np.concatenate((TargetTestData, TargetData_temp))
            TargetTestLabel = np.concatenate((TargetTestLabel, TargetLabel_temp))
            ShadowTestData = np.concatenate((ShadowTestData, ShadowData_temp))
            ShadowTestLabel = np.concatenate((ShadowTestLabel, ShadowLabel_temp))
    '''
    TargetTrainData = train_data_array[40000:50000]
    TargetTrainLabel = train_targets_array[40000:50000]
    TargetTestData = test_data_array
    TargetTestLabel = test_targets_array
    ShadowTrainData = train_data_array[:10000]
    ShadowTrainLabel = train_targets_array[:10000]
    ShadowTestData = train_data_array[10000:20000]
    ShadowTestLabel = train_targets_array[10000:20000]


    print(TargetTrainData.shape, TargetTrainLabel.shape, ShadowTrainData.shape, ShadowTrainLabel.shape)
    print(TargetTestData.shape, TargetTestLabel.shape, ShadowTestData.shape, ShadowTestLabel.shape)

    try:
        os.makedirs(dataPath + dataset)
    except OSError as e:
        pass
    np.savez(dataPath + dataset + '/targetTrain.npz', TargetTrainData, TargetTrainLabel)
    np.savez(dataPath + dataset + '/targetTest.npz',  TargetTestData, TargetTestLabel)
    np.savez(dataPath + dataset + '/shadowTrain.npz', ShadowTrainData, ShadowTrainLabel)
    np.savez(dataPath + dataset + '/shadowTest.npz',  ShadowTestData, ShadowTestLabel)

    logger.info('initializeData finished')
    print("initializeData finished\n\n")





def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
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
    parser.add_argument('--update-dataset','-u', type=int, default=0,
                        help='boolean for re-produce dataset')
    parser.add_argument('--train-target-model','-tm', type=int, default=0,
                        help='boolean for train shadow model')
    parser.add_argument('--train-shadow-model','-sm', type=int, default=0,
                        help='boolean for train shadow model')
    parser.add_argument('--ep', type=float, default=-1,
                        help='epsilon bound')
    parser.add_argument('--nm', type=float, default=-1,
                        help='noise_multiplier')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset name')
    parser.add_argument('--param', type=int, default='0',
                        help='cnn and generator parameter amount')
    parser.add_argument('--loss-name', type=str, default='CrossEntropyLoss',
                        help='loss function name')
                            
    args = parser.parse_args()
    log_path = "log/dataset_" + args.dataset + "_ep_" + str(args.ep) + "_nm_" + str(args.nm) + \
                         "_param_" + str(args.param) + "_" + str(time.time())+ "_exp.log"
    logger = util.get_logger(log_path)
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    sub_method = [False, False, True, False] ##Laplacian Smoothing , Tempered Sigmoid , Gradient Encoding, model pruning

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if(args.update_dataset):
        initializeData(args.dataset, logger)
    ## use for shokri_membership_inference

    print('--------------load data-----------------------')
    targetTrain, targetTrainLabel  = load_data(dataPath + args.dataset + '/targetTrain.npz')
    targetTest,  targetTestLabel   = load_data(dataPath + args.dataset + '/targetTest.npz')
    shadowTrain, shadowTrainLabel  = load_data(dataPath + args.dataset + '/shadowTrain.npz')
    shadowTest,  shadowTestLabel   = load_data(dataPath + args.dataset + '/shadowTest.npz')

    if args.dataset == 'mnist':
        model_name = 'ConvNet_mnist'
    elif args.dataset == 'cifar_10':
        model_name = 'ResNet18v2_cifar10'
    elif args.dataset == 'cifar_100':
        model_name = 'ConvNet_cifar100'

    #Load TARGET model
    print('--------------load model-----------------------')
    data_source=(targetTrain, targetTrainLabel, targetTest, targetTestLabel)
    #model_name,dataset,model_type,seed,nm,batchsize,epoch,optimizer,lr,loss_name = 'Target', Target_dataset, 'resnet50', args.seed, args.nm, 64, 30, 'Adam', 9e-5, 'CrossEntropyLoss'  ##nondp
    #model_name,dataset,model_type,seed,nm,batchsize,epoch,optimizer,lr,loss_name = 'Target', Target_dataset, 'ConvNet_mnist', args.seed, args.nm, 100, 200, 'Adam', 2e-3, 'CrossEntropyLoss' ##dp
    model_id = 'Target'
    model_type = 'multi_class'

    #train_param = (sub_method, dataset, True, data_source, True, model_name, model_type, 'cnn', (1,28,28), args.seed, args.nm, args.ep, batchsize, batchsize, epoch, 0, optimizer, lr, loss_name, 0.9, 10, logger, True)
    logger.info('---------------------hyperparameter-----------------------')
    logger.info('model_id: {}, model_type: {}, seed: {}, nm: {}, batchsize: {}, epoch: {}, lr: {}, loss: {}'.format(model_name,model_type,args.seed,args.nm,args.batch_size,args.epochs,args.lr,args.loss_name))
    logger.info('train/load target model')
    if args.train_target_model:
        target_model, model_dir = trainNN.train(args, sub_method, model_id, model_name, model_type, data_source, logger)
        copyfile(log_path, model_dir + '/train.log')   ## copy log to model save dir
    else:
        if args.dataset == 'cifar_100':
            target_model = model.ConvNet_cifar100(args.param).to(device)
            target_model.load_state_dict(torch.load("model/target_dataset_cifar_100_ep_-1_nm_-1_epoch_30_param_0_dataset_custom_1_model_type_cnn.pt"))
        elif args.dataset == 'mnist':
            target_model = model.ConvNet_mnist(args.param).to(device)
            target_model.load_state_dict(torch.load("model/save/exp2/Target_dataset_mnist_ep_-1_nm_-1_epoch_200_1623232049.0586293_model_pruning_98.82_0.9798/model.pt"))
        elif args.dataset == 'cifar_10':
            target_model = model.ResNet18v2_cifar10(args.param).to(device)
            target_model.load_state_dict(torch.load("model/save/exp2/Target_dataset_cifar_10_ep_10.0_nm_0.667_epoch_200_1623247957.505025_Laplacian_Smoothing_30.06_0.2682/model.pt"))
            #target_model = model.resnet50(args.param).to(device)
            #target_model.load_state_dict(torch.load("model/+Target_dataset_cifar_10_ep_-1_nm_-1_epoch_150_param_0_dataset_custom_True_model_type_cnn_1618557344.0328155_99.29_0.6997/model_0.6.pt"))

    data_source=(shadowTrain, shadowTrainLabel, shadowTest, shadowTestLabel)
    model_id = 'Shadow'
    #model_name,dataset,model_type,seed,nm,batchsize,epoch,optimizer,lr,loss_name = 'Shadow', Shadow_dataset, 'resnet101', args.seed, args.nm, 64, 5, 'Adam', 6e-6, 'CrossEntropyLoss'
    #model_name,dataset,model_type,seed,nm,batchsize,epoch,optimizer,lr,loss_name = 'Shadow', Shadow_dataset, 'ConvNet_mnist', args.seed, args.nm, 100, 200, 'Adam', 2e-3, 'CrossEntropyLoss'
    
    #train_param = (sub_method, dataset, True, data_source, True, model_name, model_type, 'cnn', (1,28,28), seed, args.nm, args.ep, batchsize, batchsize, epoch, 0, optimizer, lr, loss_name, 0.9, 10, logger, True)
    logger.info('---------------------hyperparameter-----------------------')
    logger.info('model: {}, model_type: {}, seed: {}, nm: {}, batchsize: {}, epoch: {}, lr: {}, loss: {}'.format(model_name,model_type,args.seed,args.nm,args.batch_size,args.epochs,args.lr,args.loss_name))
    logger.info('train/load shadow model')
    # train or load SHADOW model
    if args.train_shadow_model:
        shadow_model, model_dir = trainNN.train(args, sub_method, model_id, model_name, model_type, data_source, logger)
        copyfile(log_path, model_dir + '/train.log')   ## copy log to model save dir
    else:
        if args.dataset == 'cifar_100':
            shadow_model = model.ConvNet_cifar100(args.param).to(device)
            shadow_model.load_state_dict(torch.load("model/shadow_dataset_cifar_100_ep_-1_nm_-1_epoch_5_param_0_dataset_custom_1_model_type_cnn.pt"))
        elif args.dataset == 'mnist':
            shadow_model = model.ConvNet_mnist(args.param).to(device)
            shadow_model.load_state_dict(torch.load("model/save/exp2/Shadow_dataset_mnist_ep_-1_nm_-1_epoch_200_1623306818.9482195_model_pruning_98.5_0.9762/model.pt"))
        elif args.dataset == 'cifar_10':
            shadow_model = model.ResNet18v2_cifar10(args.param).to(device)
            shadow_model.load_state_dict(torch.load("model/save/Shadow_dataset_cifar_10_ep_10.0_nm_0.667_epoch_200_1623359498.9434292_Laplacian_Smoothing_30.69_0.2768/model_acc_0.2.pt"))
            #shadow_model = model.resnet101(args.param).to(device)
            #shadow_model.load_state_dict(torch.load("model/+Shadow_dataset_cifar_10_ep_-1_nm_-1_epoch_150_param_0_dataset_custom_True_model_type_cnn_1618560379.3184357_99.55_0.6964/model_0.6.pt"))
    logger.info('train/load model end')

    train_kwargs = {'batch_size': 50}
    test_kwargs = {'batch_size': 50}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    ###cal each sample predict value and loss,  y=1 for trainset and y=0 for testset
    print('--------------cal TARGET attack sample/cal TARGET train_loss,test_loss-----------------------')
    attack_x, attack_y, per_instance_loss  = [], [], []
    avg_train_loss, avg_test_loss, temp_per_instance_loss = 0, 0, []
    

    target_model.train()
    data_source=(targetTrain, targetTrainLabel, targetTest, targetTestLabel)
    train_dataset, test_dataset, _ = util.construct_dataset(args.dataset, data_source)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = target_model(data)
            train_loss = nn.CrossEntropyLoss(reduction='none')(output, target) # sum up batch loss)
            attack_x.append(output.cpu().numpy())
            attack_y.append(np.ones(data.shape[0]))
            per_instance_loss.append(train_loss.cpu().numpy())
    avg_train_loss = np.mean(np.vstack(per_instance_loss).reshape(-1,1))
    
    ## 
    target_model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = target_model(data)

            
            test_loss = nn.CrossEntropyLoss(reduction='none')(output, target) # sum up batch loss
            attack_x.append(output.cpu().numpy())
            attack_y.append(np.zeros(data.shape[0]))
            per_instance_loss.append(test_loss.cpu().numpy())
            temp_per_instance_loss.append(test_loss.cpu().numpy())
    avg_test_loss = np.mean(np.vstack(temp_per_instance_loss).reshape(-1,1))

    target_attack_x = np.vstack(attack_x).astype('float32')
    target_attack_y = np.vstack(attack_y).reshape(-1,1).astype('int32')
    target_per_instance_loss = np.vstack(per_instance_loss).reshape(-1,1).astype('float32')
    #target_classes = np.concatenate((subset_targetTrainLabel,subset_targetTestLabel)).astype('int32')

    ###cal each sample predict value and loss,  y=1 for trainset and y=0 for testset
    print('--------------cal SHADOW attack sample-----------------------')
    attack_x, attack_y, per_instance_loss  = [], [], []

    
    ## 
    shadow_model.train()
    data_source=(shadowTrain, shadowTrainLabel, shadowTest, shadowTestLabel)
    train_dataset, test_dataset, _ = util.construct_dataset(args.dataset, data_source)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = shadow_model(data)

            train_loss = nn.CrossEntropyLoss(reduction='none')(output, target) # sum up batch loss
            attack_x.append(output.cpu().numpy())
            attack_y.append(np.ones(data.shape[0]))
            per_instance_loss.append(train_loss.cpu().numpy())

    ## 
    shadow_model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = shadow_model(data)

            
            test_loss = nn.CrossEntropyLoss(reduction='none')(output, target) # sum up batch loss
            attack_x.append(output.cpu().numpy())
            attack_y.append(np.zeros(data.shape[0]))
            per_instance_loss.append(test_loss.cpu().numpy())

    shadow_attack_x = np.vstack(attack_x).astype('float32')
    shadow_attack_y = np.vstack(attack_y).reshape(-1,1).astype('int32')
    shadow_per_instance_loss = np.vstack(per_instance_loss).reshape(-1,1).astype('float32')
    #shadow_classes = np.concatenate((subset_shadowTrainLabel,subset_shadowTestLabel)).astype('int32')


    
    # clip confident vector
    '''
    topX = 10
    target_attack_x = clipDataTopX(target_attack_x,top=topX)
    shadow_attack_x = clipDataTopX(shadow_attack_x,top=topX)
    '''

    shadow_attack_x = np.sort(shadow_attack_x, axis=1)[:, ::-1]
    print(shadow_attack_x.shape)
    print(shadow_attack_x[0])
    target_attack_x = np.sort(target_attack_x, axis=1)[:, ::-1]
    print(target_attack_x.shape)
    print(target_attack_x[0])


    attack.yeom_membership_attack(shadow_per_instance_loss, shadow_attack_y, avg_train_loss, logger=logger)
    
    data_source=(shadow_attack_x, shadow_attack_y.astype('long').reshape(-1), target_attack_x, target_attack_y.astype('long').reshape(-1))
    #model_name,dataset,model_type,seed,nm,batchsize,epoch,optimizer,lr,loss_name = 'attack', Shadow_dataset, 'softmax_model', args.seed, args.nm, 64, 15, 'Adam', 1e-4, 'BCELoss'
    model_id = 'attack'
    model_type = 'binary_class'
    model_name = 'softmax_model'
    args.lr = 1e-4
    args.batch_size = 64
    args.epochs = 15
    args.loss_name = 'BCELoss'

    #train_param = (sub_method, dataset, True, data_source, True, model_name, model_type, 'softmax', None, seed, nm, args.ep, batchsize, batchsize, epoch, 0, optimizer, lr, loss_name, 0.9, 10, logger, True)
    logger.info('---------------------hyperparameter-----------------------')
    logger.info('model: {}, model_type: {}, seed: {}, nm: {}, batchsize: {}, epoch: {}, lr: {}, loss: {}'.format(model_name,model_type,args.seed,args.nm,args.batch_size,args.epochs,args.lr,args.loss_name))
    attack.shokri_membership_inference_one_shadow_model(args, sub_method, model_id, model_name, model_type, data_source, logger, log_path)






if __name__ == '__main__':
    main()
