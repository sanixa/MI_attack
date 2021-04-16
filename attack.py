from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, roc_auc_score
import scipy.stats as stats
import os
from torch.utils.data.sampler import SubsetRandomSampler 

import trainNN

dataPath = './data/'
topX = 3

def log_loss(a, b):
	return [-np.log(max(b[i,a[i]], 1e-6)) for i in range(len(a))]

def yeom_membership_inference(per_instance_loss, membership, train_loss, logger, test_loss=None):
    print('-' * 10 + 'YEOM\'S MEMBERSHIP INFERENCE' + '-' * 10 + '\n')   
    #print(stats.norm(0, train_loss).pdf(per_instance_loss))
    #print(np.e ** (-0.5*(per_instance_loss/train_loss)**2) / per_instance_loss)
    if test_loss == None:
    	pred_membership = np.where(per_instance_loss <= train_loss, 1, 0)
    else:
    	pred_membership = np.where(stats.norm(0, train_loss).pdf(per_instance_loss) >= stats.norm(0, test_loss).pdf(per_instance_loss), 1, 0)
        #pred_membership = np.where(np.e ** (-0.5*(per_instance_loss/train_loss)**2) / per_instance_loss >= np.e ** (-0.5*(per_instance_loss/test_loss)**2) / per_instance_loss, 1, 0)
    #print(membership.shape, pred_membership.shape)
    prety_print_result(membership, pred_membership, logger)
    return pred_membership

def prety_print_result(mem, pred, logger):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    logger.info(' TP: %d     FP: %d     FN: %d     TN: %d' % (tp, fp, fn, tn))
    r,p = tp/(tp+fn), tp/(tp+fp)
    logger.info(' Precision: %f     Recall: %f     F1-score: %f' % (p, r, 2*p*r/(p+r)))
    print('TP: %d     FP: %d     FN: %d     TN: %d' % (tp, fp, fn, tn))
    if tp == fp == 0:
    	print('PPV: 0\nAdvantage: 0')
    else:
    	print('PPV: %.4f\nAdvantage: %.4f' % (tp / (tp + fp), tp / (tp + fn) - fp / (tn + fp)))


def yeom_membership_attack(per_instance_loss, attack_y, train_loss, logger):

    print('\nstart eval\n')
    logger.info('start yeom_membership_attack eval')

    pred_membership = yeom_membership_inference(per_instance_loss, attack_y, train_loss, logger)
    score = roc_auc_score(attack_y, pred_membership)
    logger.info('yeom_membership_attack / roc_auc_score /' + str(score))




def shokri_membership_inference_one_shadow_model(train_param):
    print('--------------train shokri attack model-----------------------')
    '''
    ##train classifier for each class 
    train_classes, test_classes = classes
    print(train_classes.shape, test_classes.shape)
    train_indices = np.arange(len(train))
    test_indices = np.arange(len(test))
    unique_classes = np.unique(train_classes)
    print(train.shape, trainLabel.shape, test.shape, testLabel.shape)
    for c in unique_classes:
        c_train_indices = train_indices[train_classes == c]
        c_test_indices = test_indices[test_classes == c]
        print(len(c_train_indices), len(c_test_indices))
    ## attack model dataset for class c
    total_testLabel, total_pred_Label = [], []
    for c in unique_classes:
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train[c_train_indices], trainLabel[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test[c_test_indices], testLabel[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        print(c, c_train_x.shape, c_train_y.shape, c_test_x.shape, c_test_y.shape)
        model = trainNN.train(dataset=dataset, dataset_custom=1, data_source=c_dataset, model_type='softmax', epochs=30, batch_size=64, test_batch_size=64, log_interval=10)
        pred_Label = np.argmax(model(torch.tensor(c_test_x).to(torch.device("cuda"))).detach().cpu(), axis=1)

        total_testLabel.append(c_test_y)
        total_pred_Label.append(pred_Label)
    total_testLabel = np.vstack(total_testLabel).reshape(-1,1)
    total_pred_Label = np.vstack(total_pred_Label).reshape(-1,1)
    print(total_testLabel, total_pred_Label)
    prety_print_result(total_testLabel, total_pred_Label)
    print(roc_auc_score(total_testLabel, total_pred_Label))
    '''
    model = trainNN.train(train_param)

    data_source, logger = train_param[2], train_param[-2]
    train, trainLabel, test, testLabel = data_source
    #train
    pred_Label = model(torch.tensor(train.copy()).to(torch.device("cuda"))).detach().cpu() > 0.5
    prety_print_result(trainLabel, pred_Label, logger)
    #test
    pred_Label = model(torch.tensor(test.copy()).to(torch.device("cuda"))).detach().cpu() > 0.5
    prety_print_result(testLabel, pred_Label, logger)
    score = roc_auc_score(testLabel, pred_Label)
    logger.info('shokri_membership_inference_one_shadow_model / roc_auc_score /' + str(score))