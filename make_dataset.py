import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import utils.lsun_loader as lsun_loader
import utils.svhn_loader as svhn
from utils.imagenet_rc_loader import ImageNet
import pathlib
import torchvision
from dataloader.randaugment import RandAugmentMC


from PIL import Image
from torchvision import datasets

'''
This script makes the datasets used in training. The main function is make_datasets. 
'''

# *** update this before running on your machine ***
cifar10_path = './datasets/cifar10/cifarpy'
cifar100_path = './datasets/cifar-100-python'
svhn_path = './datasets/SVHN'
lsun_c_path = './datasets/LSUN_C'
lsun_r_path = './datasets/LSUN_resize'
isun_path = './datasets/iSUN'
textures_path = './datasets/texture'
places_path = './datasets/places365/'
CorCIFAR10_train = './datasets/CorCIFAR10_train'
CorCIFAR10_test = './datasets/CorCIFAR10_test'
PACS = './datasets/PACS'
ImageNet100_path = './datasets/ImageNet-100/ImageNet'
ImageNet100_C_path = './datasets/ImageNet-100/ImageNet-100-C'
iNaturalist_path = './datasets/ImageNet_OOD_dataset/iNaturalist'

def load_CIFAR(dataset, classes=[]):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset in ['cifar10']:
        print('loading CIFAR-10')
        train_data = dset.CIFAR10(
            cifar10_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            cifar10_path, train=False, transform=test_transform, download=True)

    elif dataset in ['cifar100']:
        print('loading CIFAR-100')
        train_data = dset.CIFAR100(
            cifar100_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(
            cifar100_path, train=False, transform=test_transform, download=True)

    return train_data, test_data


def load_CorCifar(dataset, cortype):

    print('loading CorCIFAR-10')
    from dataloader.corcifarLoader import CorCIFARDataset as Dataset
    train_data = Dataset('train', cortype, CorCIFAR10_train)
    test_data = Dataset('test', cortype, CorCIFAR10_test)

    return train_data, test_data


def load_PACS_id(rng, target_domain):

    print('loading PACS id')
    from dataloader.PACSidLoader import PACSidDataset as Dataset
                                     
    data = Dataset('train', target_domain)
    idx = np.array(range(len(data)))
    rng.shuffle(idx)
    train_len = int(0.7 * len(data))
    train_idx = idx[:train_len]
    test_idx = idx[train_len:]

    train_data = torch.utils.data.Subset(data, train_idx)
    test_data = torch.utils.data.Subset(data, test_idx)
                                         
    return train_data, test_data    


def load_PACS_cor(rng, target_domain):

    print('loading PACS cor')
    from dataloader.PACScorLoader import PACScorDataset as Dataset

    data = Dataset('train', target_domain)
    idx = np.array(range(len(data)))
    rng.shuffle(idx)
    train_len = int(0.7 * len(data))
    train_idx = idx[:train_len]
    test_idx = idx[train_len:]

    train_data = torch.utils.data.Subset(data, train_idx)
    test_data = torch.utils.data.Subset(data, test_idx)
    return train_data, test_data


def load_ImageNet100(dataset, rng,  id_rate):
    traindir = os.path.join(ImageNet100_path, 'train')
    valdir = os.path.join(ImageNet100_path, 'val')

    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    train_data = torchvision.datasets.ImageFolder(
        traindir,
        trn.Compose([
            trn.RandomResizedCrop(224),
            #trn.Resize(32),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ]))

    test_data = torchvision.datasets.ImageFolder(
        valdir,
        trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ]))

    idx = np.array(range(len(train_data)))
    rng.shuffle(idx)

    train_len = int(id_rate * len(train_data))
    train_idx = idx[:train_len]
    aux_idx = idx[train_len:]
    train_in_data = torch.utils.data.Subset(train_data, train_idx)
    aux_in_data = torch.utils.data.Subset(train_data, aux_idx)

    return train_in_data, aux_in_data, test_data

def load_CorImageNet100(dataset, cortype):

    from dataloader.corimagenetLoader import CorIMAGENETDataset as Dataset
    train_data = Dataset('train', cortype)
    test_data = Dataset('test', cortype)

    return train_data, test_data



def load_SVHN(include_extra=False):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    print('loading SVHN')
    if not include_extra:
        train_data = svhn.SVHN(root=svhn_path, split="train",
                                 transform=trn.Compose(
                                         trn.ToTensor(), trn.Normalize(mean, std)]))
    else:
        train_data = svhn.SVHN(root=svhn_path, split="train_and_extra",
                               transform=trn.Compose(
                                       trn.ToTensor(), trn.Normalize(mean, std)]))

    test_data = svhn.SVHN(root=svhn_path, split="test",
                              transform=trn.Compose(
                                      trn.ToTensor(), trn.Normalize(mean, std)]))

    train_data.targets = train_data.targets.astype('int64')
    test_data.targets = test_data.targets.astype('int64')
    return train_data, test_data


def load_dataset(dataset):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if dataset == 'lsun_c':
        print('loading LSUN_C')
        out_data = dset.ImageFolder(root=lsun_c_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std),
                                                           trn.RandomCrop(32, padding=4)]))

    if dataset == 'lsun_r':
        print('loading LSUN_R')
        out_data = dset.ImageFolder(root=lsun_r_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

    if dataset == 'isun':
        print('loading iSUN')
        out_data = dset.ImageFolder(root=isun_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    if dataset == 'dtd':
        print('loading DTD')
        out_data = dset.ImageFolder(root=textures_path,
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
       
    if dataset == 'places':
        print('loading Places365')
        out_data = dset.ImageFolder(root=places_path,
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))

    if dataset == 'iNaturalist':
        print('iNaturalist')
        out_data = dset.ImageFolder(root=iNaturalist_path, transform=trn.Compose([
                                trn.Resize(256),
                                trn.CenterCrop(224), 
                                trn.ToTensor(),
                                trn.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        )      

    return out_data


def load_in_data(in_dset, rng, id_rate, target_domain='cartoon'): 

    if in_dset in ['cifar10']:
        train_data_in_orig, test_in_data = load_CIFAR(in_dset) 
    else:
        train_data_in_orig, test_in_data = load_PACS_id(rng, target_domain)   

    idx = np.array(range(len(train_data_in_orig)))
    rng.shuffle(idx)
    train_len = int(id_rate * len(train_data_in_orig))
    train_idx = idx[:train_len]
    aux_idx = idx[train_len:]

    train_in_data = torch.utils.data.Subset(train_data_in_orig, train_idx)
    aux_in_data = torch.utils.data.Subset(train_data_in_orig, aux_idx)

    return train_in_data, aux_in_data, test_in_data


def load_cor_data(in_dset, rng, cortype, target_domain):

    if in_dset in ['cifar10']:
        aux_data_cor_orig, test_cor_data = load_CorCifar(in_dset, cortype) 
    else:
        aux_data_cor_orig, test_cor_data = load_PACS_cor(rng, target_domain) 
    return aux_data_cor_orig, test_cor_data 


def load_out_data(aux_out_dset, test_out_dset, in_dset, rng, classes=[]):
    if aux_out_dset == test_out_dset:

        if aux_out_dset == 'svhn':
            aux_out_data, test_out_data = load_SVHN()
        elif aux_out_dset == 'cifar100':
            aux_out_data, test_out_data = load_CIFAR(aux_out_dset)
        else:
            out_data = load_dataset(aux_out_dset)

            idx = np.array(range(len(out_data)))
            rng.shuffle(idx)
            train_len = int(0.7 * len(out_data))
            aux_subset_idxs = idx[:train_len]
            test_subset_idxs = idx[train_len:]

            aux_out_data = torch.utils.data.Subset(out_data, aux_subset_idxs)
            test_out_data = torch.utils.data.Subset(out_data, test_subset_idxs)
               

    elif aux_out_dset != test_out_dset:
        # load aux data
        if aux_out_dset == 'svhn':
            aux_out_data, _ = load_SVHN()
        elif aux_out_dset in ['cifar10', 'cifar100']:
            aux_out_data, _ = load_CIFAR(aux_out_dset)
        else:
            aux_out_data = load_dataset(aux_out_dset)

        # load test data
        if test_out_dset == 'svhn':
            _, test_out_data = load_SVHN()
        elif test_out_dset in ['cifar10', 'cifar100']:
            _, test_out_data = load_CIFAR(test_out_dset)
        else:
            test_out_data = load_dataset(test_out_dset)

    return aux_out_data, test_out_data


def train_valid_split(aux_in_data, aux_cor_data, aux_out_data, rng, ood_rate_1, ood_rate_2):
    '''
    Args:
        aux_in_data: data from in-distribution component of mixture, not in test set
        aux_cor_data: data from auxiliary dataset component of mixture, not in test set      
        aux_out_data: data from auxiliary dataset component of mixture, not in test set
    Returns:
        6 datasets: each dataset split into two, one for training and the other for validation
    '''

    aux_in_valid_size = int(0.4 * len(aux_in_data))

    idx = np.array(range(len(aux_in_data)))
    rng.shuffle(idx)
    train_in_idx = idx[aux_in_valid_size:]
    valid_in_idx = idx[:aux_in_valid_size]

    train_aux_in_final = torch.utils.data.Subset(aux_in_data, train_in_idx)
    valid_aux_in_final = torch.utils.data.Subset(aux_in_data, valid_in_idx)

    aux_cor_valid_size = int(0.4 * len(aux_cor_data))

    idx = np.array(range(len(aux_cor_data)))
    rng.shuffle(idx)
    train_cor_idx = idx[aux_cor_valid_size:]

    valid_cor_idx = idx[:aux_cor_valid_size]

    train_aux_cor_final = torch.utils.data.Subset(aux_cor_data, train_cor_idx)
    valid_aux_cor_final = torch.utils.data.Subset(aux_cor_data, valid_cor_idx)

    aux_out_valid_size = int(0.3 * len(aux_out_data))

    idx = np.array(range(len(aux_out_data)))
    rng.shuffle(idx)
    train_out_idx = idx[aux_out_valid_size:]
    valid_out_idx = idx[:aux_out_valid_size]

    train_aux_ood_final = torch.utils.data.Subset(aux_out_data, train_out_idx)
    valid_aux_ood_final = torch.utils.data.Subset(aux_out_data, valid_out_idx)

    return train_aux_in_final, train_aux_cor_final, train_aux_ood_final, valid_aux_in_final, valid_aux_cor_final, valid_aux_ood_final


def make_datasets(in_dset, aux_out_dset, test_out_dset, ood_rate_1, ood_rate_2, state, cortype, id_rate, target_domain):
    # random seed
    rng = np.random.default_rng(state['seed'])
    print('building datasets...')

    if in_dset in ['cifar10', 'cifar100']:
        train_in_data, aux_in_data, test_in_data = load_in_data(in_dset, rng, id_rate) 
        aux_cor_data, test_cor_data = load_cor_data(in_dset, rng, cortype) 
        
    elif in_dset in ['PACS']:
        train_in_data, aux_in_data, test_in_data = load_in_data(in_dset, rng, id_rate, target_domain) 
        aux_cor_data, test_cor_data = load_cor_data(in_dset, rng, cortype, target_domain) 

    else:
        train_in_data, aux_in_data, test_in_data = load_ImageNet100(in_dset, rng, id_rate) 
        aux_cor_data, test_cor_data = load_CorImageNet100(in_dset, cortype) 

    aux_out_data, test_out_data = load_out_data(aux_out_dset,
                                                test_out_dset, in_dset, rng)

    train_aux_in_data_final, train_aux_cor_data_final, train_aux_ood_data_final,\
         valid_aux_in_final, valid_aux_cor_final, valid_aux_ood_final = train_valid_split(
                                aux_in_data, aux_cor_data, aux_out_data, rng, ood_rate_1, ood_rate_2
                            )


    train_loader_in = torch.utils.data.DataLoader(
        train_in_data,
        batch_size=state['batch_size'], shuffle=True, 
        num_workers=state['prefetch'], pin_memory=True)

    train_loader_in_large_bs = torch.utils.data.DataLoader(
        train_in_data,
        batch_size=1280, shuffle=True,
        num_workers=state['prefetch'], pin_memory=True)
    #for in-distribution component of mixture
    #drop last batch to eliminate unequal batch size issues

    train_loader_aux_in = torch.utils.data.DataLoader(
        train_aux_in_data_final,
        batch_size=int(120*(1-ood_rate_1-ood_rate_2)), shuffle=True, 
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    train_loader_aux_cor = torch.utils.data.DataLoader(
        train_aux_cor_data_final,
        batch_size=int(120*ood_rate_1), shuffle=True, 
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    train_loader_aux_out = torch.utils.data.DataLoader(
        train_aux_ood_data_final,
        batch_size=int(120*ood_rate_2), shuffle=True, 
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    # test data for P_0
    test_loader = torch.utils.data.DataLoader(
        test_in_data,
        batch_size=128, shuffle=False, 
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    # test loader for covariate-shifted ood
    test_loader_cor = torch.utils.data.DataLoader(
        test_cor_data,
        batch_size=128, shuffle=False, 
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    # test loader for ood
    test_loader_out = torch.utils.data.DataLoader(
        test_out_data,
        batch_size=128, shuffle=False, 
        num_workers=state['prefetch'], pin_memory=True, drop_last=True)

    return train_loader_in, train_loader_in_large_bs, train_loader_aux_in, \
           train_loader_aux_cor, train_loader_aux_out, test_loader, \
           test_loader_cor, test_loader_out 

