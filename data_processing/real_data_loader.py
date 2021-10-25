import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shelve
import pickle
import os
from sklearn.preprocessing import StandardScaler

def split_real_data_into_folds(d):
    '''
    Splits data into 4 folds, stratified by provider.
    For each provider 50% of data is in test.
    Among remaining 50%, around 75% are used for training and around 25% are used for validation in each fold.
    Validation splits only overlap between folds if fewer than 4 samples for that provider in training/validation split.
    
    Parameters
    ----------
    d: numpy array
        List of agents for each sample
        
    Returns
    -------
    train_fold_idxs: dictionary
        Map fold index to indices of training samples
        
    valid_fold_idxs: dictionary
        Map fold index to indices of validation samples
    
    test_idxs: numpy array
        Indices of test samples in all folds
    '''
    np.random.seed(1780)
    train_fold_idxs = dict()
    valid_fold_idxs = dict()
    for i in range(int(np.max(d)+1)):
        prov_idxs = np.nonzero(np.where(d == i, 1, 0))[0]
        np.random.shuffle(prov_idxs)
        prov_train_valid_idxs = prov_idxs[:int(.5*len(prov_idxs))]
        prov_train_valid_fold_idxs = dict()
        for fold_idx in range(4):
            prov_train_valid_fold_idxs[fold_idx] = dict()
        assert len(prov_train_valid_idxs) >= 2
        if len(prov_train_valid_idxs) == 2:
            prov_train_valid_fold_idxs[0]['train'] = prov_train_valid_idxs[:1]
            prov_train_valid_fold_idxs[0]['valid'] = prov_train_valid_idxs[1:]
            prov_train_valid_fold_idxs[1]['train'] = prov_train_valid_idxs[:1]
            prov_train_valid_fold_idxs[1]['valid'] = prov_train_valid_idxs[1:]
            prov_train_valid_fold_idxs[2]['train'] = prov_train_valid_idxs[1:]
            prov_train_valid_fold_idxs[2]['valid'] = prov_train_valid_idxs[:1]
            prov_train_valid_fold_idxs[3]['train'] = prov_train_valid_idxs[1:]
            prov_train_valid_fold_idxs[3]['valid'] = prov_train_valid_idxs[:1]
        elif len(prov_train_valid_idxs) == 3:
            prov_train_valid_fold_idxs[0]['train'] = prov_train_valid_idxs[:2]
            prov_train_valid_fold_idxs[0]['valid'] = prov_train_valid_idxs[2:]
            prov_train_valid_fold_idxs[1]['train'] = prov_train_valid_idxs[:2]
            prov_train_valid_fold_idxs[1]['valid'] = prov_train_valid_idxs[2:]
            prov_train_valid_fold_idxs[2]['train'] = prov_train_valid_idxs[1:]
            prov_train_valid_fold_idxs[2]['valid'] = prov_train_valid_idxs[:1]
            prov_train_valid_fold_idxs[3]['train'] = prov_train_valid_idxs[[0,2]]
            prov_train_valid_fold_idxs[3]['valid'] = prov_train_valid_idxs[1:2]
        else:
            prov_train_valid_fold_idxs[0]['train'] = prov_train_valid_idxs[:int(.75*len(prov_train_valid_idxs))]
            prov_train_valid_fold_idxs[0]['valid'] = prov_train_valid_idxs[int(.75*len(prov_train_valid_idxs)):]
            prov_train_valid_fold_idxs[1]['train'] \
                = np.concatenate((prov_train_valid_idxs[:int(.5*len(prov_train_valid_idxs))], 
                                  prov_train_valid_idxs[int(.75*len(prov_train_valid_idxs)):]))
            prov_train_valid_fold_idxs[1]['valid'] \
                = prov_train_valid_idxs[int(.5*len(prov_train_valid_idxs)):int(.75*len(prov_train_valid_idxs))]
            prov_train_valid_fold_idxs[2]['train'] = \
                np.concatenate((prov_train_valid_idxs[:int(.25*len(prov_train_valid_idxs))], 
                                prov_train_valid_idxs[int(.5*len(prov_train_valid_idxs)):]))
            prov_train_valid_fold_idxs[2]['valid'] \
                = prov_train_valid_idxs[int(.25*len(prov_train_valid_idxs)):int(.5*len(prov_train_valid_idxs))]
            prov_train_valid_fold_idxs[3]['train'] = prov_train_valid_idxs[int(.25*len(prov_train_valid_idxs)):]
            prov_train_valid_fold_idxs[3]['valid'] = prov_train_valid_idxs[:int(.25*len(prov_train_valid_idxs))]
        prov_test_idxs = prov_idxs[int(.5*len(prov_idxs)):] # int() guarantees at most len(prov_idxs)-1, so >= 1 sample
        if i == 0:
            for fold_idx in range(4):
                train_fold_idxs[fold_idx] = prov_train_valid_fold_idxs[fold_idx]['train']
                valid_fold_idxs[fold_idx] = prov_train_valid_fold_idxs[fold_idx]['valid']
            test_idxs = prov_test_idxs
        else:
            for fold_idx in range(4):
                train_fold_idxs[fold_idx] = np.concatenate((train_fold_idxs[fold_idx], 
                                                            prov_train_valid_fold_idxs[fold_idx]['train']))
                valid_fold_idxs[fold_idx] = np.concatenate((valid_fold_idxs[fold_idx], 
                                                            prov_train_valid_fold_idxs[fold_idx]['valid']))
            test_idxs = np.concatenate((test_idxs, prov_test_idxs))
    return train_fold_idxs, valid_fold_idxs, test_idxs

def load_diabetes_data():
    '''
    Load first-line diabetes dataset.
    
    Returns
    -------
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
        
    train_fold_idxs: dictionary
        Indices in training set of each fold
        
    valid_fold_idxs: list of ints
        Indices in validation set of each fold
        
    test_idxs: list of ints
        Indices in test set
    
    orig_prvs: list of str
        Names of agents
    
    scaler: sklearn StandardScaler object
        Used to scale X
    '''
    PATH_TO_DATA = # TODO WHERE PROPRIETARY DATA IS LOCATED
    df = pd.read_pickle(PATH_TO_DATA)
    prv_counts = df['prv'].value_counts().sort_index() # Sorting to make deterministic
    prv_counter = 0
    prv_dict = dict()
    orig_prvs = []
    for prv in prv_counts.keys():
        if prv_counts[prv] >= 4:
            prv_dict[prv] = prv_counter
            prv_counter += 1
            orig_prvs.append(prv)
    df = df[df['prv'].isin(set(list(prv_dict.keys())))]

    X = df[['egfr','creatinine','heart_disease','treatment_date_sec']].values
    d_orig = df['prv'].values
    d = np.empty(d_orig.shape)
    for i in range(len(d_orig)):
        d[i] = prv_dict[d_orig[i]]
    d = d.astype(int)
    t = df['y'].values
    train_fold_idxs, valid_fold_idxs, test_idxs = split_real_data_into_folds(d)
    train_valid_idxs = np.concatenate((train_fold_idxs[0], valid_fold_idxs[0]))
    train_valid_X = X[train_valid_idxs]
    scaler = StandardScaler()
    scaler.fit(train_valid_X)
    X = scaler.transform(X)
    return X, d, t, train_fold_idxs, valid_fold_idxs, test_idxs, orig_prvs, scaler

def load_ppmi_data():
    '''
    Load Parkinson's dataset. See load_diabetes_data for return values
    '''
    PATH_TO_DATA = #TODO WHERE YOU PUT THE PPMI DATA (csv with 1st line treatment, age, mds 2+3 score, disease duration, site)
    df = pd.read_csv(PATH_TO_DATA)
    site_counts = df['site'].value_counts().sort_index() # Sorting to make deterministic
    site_counter = 0
    site_dict = dict()
    orig_sites = [] # sort keys below to make deterministic
    for site in site_counts.keys():
        if site_counts[site] >= 4:
            site_dict[site] = site_counter
            site_counter += 1
            orig_sites.append(site)
    df = df[df['site'].isin(set(list(site_dict.keys())))]
    X = df[['age','disdur','mds23']].values
    d_orig = df['site'].values
    d = np.empty(d_orig.shape)
    for i in range(len(d_orig)):
        d[i] = site_dict[d_orig[i]]
    d = d.astype(int)
    t = df['treatment'].values
    train_fold_idxs, valid_fold_idxs, test_idxs = split_real_data_into_folds(d)
    train_valid_idxs = np.concatenate((train_fold_idxs[0], valid_fold_idxs[0]))
    train_valid_X = X[train_valid_idxs]
    scaler = StandardScaler()
    scaler.fit(train_valid_X)
    X = scaler.transform(X)
    return X, d, t, train_fold_idxs, valid_fold_idxs, test_idxs, orig_sites, scaler
