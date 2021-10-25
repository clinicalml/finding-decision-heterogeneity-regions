import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from itertools import product
from econml.grf import CausalForest
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from helpers import (
    train_logreg, 
    train_linear_reg, 
    train_decision_tree, 
    train_decision_tree_reg, 
    train_random_forest, 
    train_random_forest_reg,
    compute_error_diff,
    compute_diff_perc_cutoff,
    eval_region_precision_recall_auc,
    compute_logistic_provider_split,
    eval_provider_split_acc
)


'''
Algorithms to identify regions of disagreement.
'''


def get_outcome_model(secondary_class, X_train, X_valid, t_train, t_valid):
    '''
    Trains a model of a specified class using cross-validation.

    Parameters
    ----------
    secondary_class : str
        A string specifying the type of model to be trained. Supports 
        ['LogisticRegression', 'DecisionTree', 'RandomForest']

    X_train : array-like of shape (n_train_samples, n_features)
        Training vector, where n_train_samples is the number of training samples
        and n_features is the number of features.

    X_valid : array-like of shape (n_valid_samples, n_features)
        Validation vector, where n_valid samples is the number of validation samples.

    t_train : array-like of shape (n_train_samples,)
        Target vector relative to X_train.

    t_valid : array-like of shape (n_valid_samples,)
        Target vector relative to X_valid.

    Returns
    -------
    outcome_model
        A model of the specified class.

    '''
    assert secondary_class in [
            'LogisticRegression',
            'DecisionTree',
            'RandomForest'], f'Unsupported model class {secondary_class}'

    if secondary_class == 'LogisticRegression':
        outcome_model = train_logreg(X_train, t_train, X_valid, t_valid)
    elif secondary_class == 'DecisionTree':
        if X_train.shape[0] > 10000:
            min_samples_leaf_options = [100, 500, 1000, 5000, 10000]
        else:
            min_samples_leaf_options = [10, 25, 100]
        outcome_model = train_decision_tree(X_train, t_train, X_valid, t_valid, min_samples_leaf_options)
    else: # secondary_class == 'RandomForest'
        outcome_model = train_random_forest(X_train, t_train, X_valid, t_valid)

    return outcome_model

def get_and_save_results(model_class, secondary_class,
                train_diff, valid_diff, test_diff, 
                X_train, t_train, d_train, 
                X_valid, t_valid, d_valid, 
                X_test, t_test, d_test, 
                Xt_train_pred, Xt_valid_pred, Xt_test_pred, 
                filename,
                n_prov=None, 
                iter_q_scores=None, iter_region_aucs=None, iter_partition_accs=None,
                region_model=None, cutoff=None,
                pred_provider_split=None, true_provider_split=None,
                true_region_func=None):
    '''
    Processes algorithm output and writes to file.

    Parameters
    ----------
    model_class : str
        A string indicating the type of model used to find the region.

    secondary_class : str
        A string indicating the type of model used to parameterize either
        the outcome_model or the region_model.

    train_diff : array-like of size (n_train_samples,)
        The outputs of a region model on the training set.

    valid_diff : array-like of size (n_valid_samples,)
        The outputs of a region model on the validation set.
        
    test_diff : array-like of size (n_test_samples,)
        The outputs of a region model on the test set.

    Xt_train_pred : array-like of shape (n_train_samples,)
        The outputs of an outcome model on the training set.

    Xt_valid_pred : array-like of shape (n_valid_samples,)
        The outputs of an outcome model on the validation set.

    Xt_test_pred : array-like of shape (n_test_samples,)
        The outputs of an outcome model on the test set.

    filename : str
        Where to write results to.

    iter_q_scores : array-like, default=None
        Q(S, G) for each iteration of the algorithm.

    iter_region_aucs : array-like, default=None
        AUC of recovering the region for each iteration.

    iter_partition_accs : array-like, default=None
        Agent group accuracy for each iteration.

    region_model, default=None
        A fitted model of the type indicated by model_class.

    cutoff : float, default=None
        Defines the region to include x such that region_model predicts
        a value above cutoff.

    pred_provider_split : array-like of shape (n_prov,), default=None
        Estimated provider groupings.

    true_provider_split : array-like of shape (n_prov,), default=None
        True provider groupings.

    true_region_func
        Either a function or a list of two functions. If a function, returns 1 if 
        input is in the region of disagreement and 0 otherwise. If a list of functions,
        each function describes a region of disagreement.
    
    Returns
    -------
    results_dict : dictionary
        Dictionary with various results, including all parameters, and:
        - region_precision, region_recall, region_auc
            Region performance metrics

        - partition_acc, test_partition_acc
            Group performance metrics

        - train_scores, valid_scores, test_scores
            When groupings are estimated, stores the numbers (Y-f(X))*G
    '''

    # Indices of data points in region
    train_region_idxs = np.nonzero(np.where(train_diff >= cutoff, 1, 0))[0]
    valid_region_idxs = np.nonzero(np.where(valid_diff >= cutoff, 1, 0))[0]
    test_region_idxs = np.nonzero(np.where(test_diff >= cutoff, 1, 0))[0]

    # If provider split was estimated, compute Q(S, G)
    if pred_provider_split is not None:
        g_train = np.array([1 if pred_provider_split[d_train[i]]==1 else 0 for i in range(len(d_train))])
        train_scores = (t_train - Xt_train_pred) * g_train
        g_valid = np.array([1 if pred_provider_split[d_valid[i]]==1 else 0 for i in range(len(d_valid))])
        valid_scores = (t_valid - Xt_valid_pred) * g_valid
      
        # Compute G specifically for the validation and test set.
        pred_provider_split_test, _ = best_G(test_region_idxs, X_test, t_test, d_test, Xt_test_pred, n_prov)
        g_test = np.array([1 if pred_provider_split_test[d_test[i]]==1 else 0 for i in range(len(d_test))])
        test_scores = (t_test - Xt_test_pred) * g_test
    else:
        pred_provider_split_test = None
        train_scores, valid_scores, test_scores = None, None, None
    
    # Performance metrics
    region_precision = None
    region_recall = None
    region_auc = None
    partition_acc = None
    test_partition_acc = None

    if true_region_func is not None:
        region_precision, region_recall, region_auc \
            = eval_region_precision_recall_auc(test_diff, cutoff, X_test, true_region_func)

    if true_provider_split is not None and pred_provider_split is not None and pred_provider_split_test is not None:
        partition_acc = eval_provider_split_acc(pred_provider_split, true_provider_split)
        test_partition_acc = eval_provider_split_acc(pred_provider_split_test, true_provider_split)

    results_dict = {'Xt_train_pred': Xt_train_pred, 'Xt_valid_pred': Xt_valid_pred, 'Xt_test_pred': Xt_test_pred, 
                    'region_precision': region_precision, 'region_recall': region_recall, 'region_auc': region_auc, 
                    'partition_acc': partition_acc, 'test_partition_acc': test_partition_acc,
                    'pred_provider_split': pred_provider_split, 'pred_provider_split_test': pred_provider_split_test,
                    'train_scores': train_scores, 'valid_scores': valid_scores, 'test_scores': test_scores, 
                    'train_region_idxs': train_region_idxs, 'valid_region_idxs': valid_region_idxs, 
                    'test_region_idxs': test_region_idxs, 'region_model': region_model, 'cutoff': cutoff, 'n_prov': n_prov,
                    'iter_q_scores': iter_q_scores, 'iter_region_aucs': iter_region_aucs, 
                    'iter_partition_accs': iter_partition_accs}

    if filename is not None:
        with open(filename, 'wb') as f:
            pickle.dump(results_dict, f)

    return results_dict

def best_G(S, X, y, a, preds, n_prv):
    '''
    Identifies the best grouping given a region. For documentation on parameters,
    see best_S_and_G().

    Parameters
    ----------
    S : array-like of shape (n_samples,)
        A boolean array indicating membership

    Returns
    -------
    G : array-like of shape (n_prv,)
        An array of 0 and 1 indicating group membership.

    score : float
        The Q(S, G) for the identified region and grouping.
    '''
    G = np.zeros(n_prv)
    score = 0.0
    for i in range(n_prv):
        ixs = (a[S] == i)
        if np.sum(ixs) > 0:
            term = (1/np.sum(S)) * np.sum(y[S][ixs] - preds[S][ixs])
            if term >= 0:
                G[i] = 1
                score += term
    return G, score

def best_S(model_class, G, beta, X, y, a, preds):
    '''
    Identifies the best region given a grouping. For documentation on parameters,
    see best_S_and_G().

    Parameters
    ----------
    G : array-like of shape (n_prv,)
        An array of 0 and 1 indicating group membership.

    Returns
    -------
    region_model
        A fitted model of the type indicated by model_class.

    cutoff : float
        Defines the region to include x such that region_model predicts
        a value above cutoff.
    '''
    
    # Compute G
    g = np.array([1 if G[a[i]]==1 else 0 for i in range(len(a))])
    scores = (y - preds) * g
    
    X_train_1, X_train_2, train_diff_1, train_diff_2 = train_test_split(X, scores, test_size=0.3, random_state=1)
    
    if model_class == 'LogisticRegression':
        region_model = train_linear_reg(X_train_1, train_diff_1, X_train_2, train_diff_2)
    elif model_class == 'DecisionTree':
        if X.shape[0] > 4000:
            min_samples_leaf_options = [100, 500, 1000]
        else:
            min_samples_leaf_options = [10, 25, 100]
        region_model = train_decision_tree_reg(X_train_1, train_diff_1, X_train_2, train_diff_2, min_samples_leaf_options)
    else: # model_class == 'RandomForest':
        region_model = train_random_forest_reg(X_train_1, train_diff_1, X_train_2, train_diff_2)
        
    scores = region_model.predict(X)
    cutoff = np.quantile(scores, 1-beta)
    
    return region_model, cutoff

def best_S_and_G(model_class, beta, X, y, a, preds, n_prv, n_iter=5, true_region_func=None, true_provider_split=None):
    '''
    Wrapper method to run the iterative update algorithm.

    Parameters
    ----------
    model_class : str
        The hypothesis class for the region. One of
        ['LogisticRegression', 'DecisionTree', 'RandomForest'].

    beta : float
        Float between 0 and 1 indicating the size of the desired
        region of disagreement.

    X, y, a : array-like
        Data inherited from run_iterative_alg().

    preds : array-like of shape (n_samples,)
        Predictions from an outcome model.

    n_prv : int
        Number of agents.

    n_iter : int
        Number of iterations.

    Returns
    -------
    S : array-like of shape (n_samples,)
        A boolean array indicating membership

    G : array-like of shape (n_prv,)
        An array of 0 and 1 indicating group membership.

    region_model
        A fitted model of the type indicated by model_class.

    cutoff : float
        Defines the region to include x such that region_model predicts
        a value above cutoff.

    iter_q_scores : array-like
        Q(S, G) for each iteration of the algorithm.

    iter_region_aucs : array-like
        AUC of recovering the region for each iteration.

    iter_partition_accs : array-like
        Agent group accuracy for each iteration.
    '''
    
    # Initialize S to the entire space.
    S = np.array([True] * X.shape[0])
    G = None
    G_prev = None
    scores = None
    cutoff = None
    
    # Store performance metrics for each iteration.
    iter_region_aucs = []
    iter_partition_accs = []
    iter_q_scores = []

    for it in range(n_iter):

        # Find the best G for the current S.
        G, iter_q_score = best_G(S, X, y, a, preds, n_prv)
        
        # Compute performance metrics.
        if true_region_func is not None and scores is not None and cutoff is not None:
            _, _, iter_region_auc = eval_region_precision_recall_auc(scores, cutoff, X, true_region_func)
            iter_region_aucs.append(iter_region_auc)
        if true_provider_split is not None and G is not None:
            iter_partition_acc = eval_provider_split_acc(G, true_provider_split)
            iter_partition_accs.append(iter_partition_acc)
        iter_q_scores.append(iter_q_score)
        
        # If G doesn't change, terminate.
        if G_prev is not None and np.all(G_prev == G):
            print('Terminated early at ' + str(it) + ' iterations')
            break
        G_prev = G

        # Find the best S for the current G.
        region_model, cutoff = best_S(model_class, G, beta, X, y, a, preds)
        scores = region_model.predict(X)
        S = scores >= cutoff
    
    # Update G and performance metrics before returning.
    G, iter_q_score = best_G(S, X, y, a, preds, n_prv)
    iter_q_scores.append(iter_q_score)
    if true_region_func is not None and scores is not None and cutoff is not None:
        _, _, iter_region_auc = eval_region_precision_recall_auc(scores, cutoff, X, true_region_func)
        iter_region_aucs.append(iter_region_auc)
    if true_provider_split is not None and G is not None:
        iter_partition_acc = eval_provider_split_acc(G, true_provider_split)
        iter_partition_accs.append(iter_partition_acc)
        
    return S, G, region_model, cutoff, iter_q_scores, iter_region_aucs, iter_partition_accs

def run_iterative_alg(secondary_class,
                      X_train, X_valid, X_test,
                      d_train, d_valid, d_test,
                      t_train, t_valid, t_test,
                      n_prov,
                      outcome_model,
                      filename, 
                      true_region_func=None, true_provider_split=None, 
                      beta=0.1, n_iter=5, region_X_feat_idxs=None):
    '''
    Run the IterativeAlg algorithm on data.

    Parameters
    ----------
    model_class : str
        A string indicating the type of model used to find the region.

    secondary_class : str
        A string indicating the type of model used to parameterize either
        the outcome_model or the region_model.

    X_train : array-like of size (n_train_samples, n_features)
        Training vector, where n_train_samples is the number of training samples
        and n_features is the number of features.

    t_train : array-like of shape (n_train_samples,)
        Target vector relative to X_train.

    d_train : array-like of shape (n_train_samples,)
        Vector of decision-makers/agents relative to X_train.

    X_valid : array-like of shape (n_valid_samples, n_features)
        Validation vector, where n_valid_samples is the number of validation samples.

    t_valid : array-like of shape (n_valid_samples,)
        Target vector relative to X_valid.

    d_valid : array-like of shape (n_valid_samples,)
        Vector of decision-makers/agents relative to X_valid.

    X_test : array-like of shape (n_test_samples, n_features)
        Test vector, where n_test_samples is the number of test samples.

    t_test : array-like of shape (n_test_samples,)
        Target vector relative to X_test.

    d_test : array-like of shape (n_test_samples,)
        Vector of decision-makers/agents relative to X_test.

    n_prov : int
        The number of unique agents/decision-makers.

    outcome_model : BaseEstimator
        A predictive model with the standard .predict_proba(X) method.

    filename : str
        The path to which results should be written.

    true_region_func
        Either a function or a list of two functions. If a function, returns 1 if 
        input is in the region of disagreement and 0 otherwise. If a list of functions,
        each function describes a region of disagreement.

    true_provider_split : array-like of shape (n_prov,)
        An array showing the groups of each agent.

    beta : float
        Size of the desired region of disagreement.

    n_iter : int
        Number of iterations to run IterativeAlg for.
        
    region_X_feat_idxs: list of ints, default: None
        Subset of feature indices used to learn region model. If None, use all features.

    Returns
    -------
    result_dict : dictionary
        A results dictionary. See the documentation for get_and_save_results().
    '''

    assert secondary_class in {'LogisticRegression', 'DecisionTree', 'RandomForest'}
    if region_X_feat_idxs is None:
        region_X_feat_idxs = range(X_train.shape[1])
    else:
        assert np.min(region_X_feat_idxs) >= 0
        assert np.max(region_X_feat_idxs) < X_train.shape[1]
        assert len(np.unique(region_X_feat_idxs)) == len(region_X_feat_idxs)
        
    # Get predictions from outcome model
    Xt_train_pred = outcome_model.predict_proba(X_train)[:,1]
    Xt_valid_pred = outcome_model.predict_proba(X_valid)[:,1]
    Xt_test_pred = outcome_model.predict_proba(X_test)[:,1]

    X_train_valid = np.vstack((X_train, X_valid))
    d_train_valid = np.concatenate((d_train, d_valid))
    t_train_valid = np.concatenate((t_train, t_valid))
    Xt_train_valid_pred = np.concatenate((Xt_train_pred, Xt_valid_pred))
    
    # Run iterative algorithm
    S, pred_provider_split, region_model, cutoff, iter_q_scores, iter_region_aucs, \
        iter_partition_accs = best_S_and_G(secondary_class, beta, X_train_valid, t_train_valid, d_train_valid, 
                                           Xt_train_valid_pred, n_prov, n_iter=n_iter, 
                                           true_region_func=true_region_func, true_provider_split=true_provider_split)
        
    train_diff = region_model.predict(X_train[:,region_X_feat_idxs])
    valid_diff = region_model.predict(X_valid[:,region_X_feat_idxs])
    test_diff = region_model.predict(X_test[:,region_X_feat_idxs])

    return get_and_save_results('IterativeAlg', secondary_class,
                                train_diff, valid_diff, test_diff, 
                                X_train, t_train, d_train, 
                                X_valid, t_valid, d_valid, 
                                X_test, t_test, d_test, 
                                Xt_train_pred, Xt_valid_pred, Xt_test_pred, 
                                filename, n_prov=n_prov, 
                                iter_q_scores=iter_q_scores, iter_region_aucs=iter_region_aucs, 
                                iter_partition_accs=iter_partition_accs,
                                region_model=region_model, cutoff=cutoff,
                                pred_provider_split=pred_provider_split,
                                true_region_func=true_region_func,
                                true_provider_split=true_provider_split)

def run_iterative_alg_tune_beta(secondary_class,
                                X_train, X_valid, X_test,
                                d_train, d_valid, d_test,
                                t_train, t_valid, t_test,
                                n_prov,
                                outcome_model,
                                filename, 
                                plot_true_region=None, plot_filename=None, 
                                true_region_func=None, true_provider_split=None, 
                                betas=np.linspace(0.02, 0.42, 11), n_iter=5, n_shuffles=40, region_X_feat_idxs=None):
    '''
    Run the IterativeAlg algorithm on data.

    Parameters
    ----------
    secondary_class : str
        A string indicating the type of model used to parameterize either
        the outcome_model or the region_model.

    X_train : array-like of size (n_train_samples, n_features)
        Training vector, where n_train_samples is the number of training samples
        and n_features is the number of features.

    t_train : array-like of shape (n_train_samples,)
        Target vector relative to X_train.

    d_train : array-like of shape (n_train_samples,)
        Vector of decision-makers/agents relative to X_train.

    X_valid : array-like of shape (n_valid_samples, n_features)
        Validation vector, where n_valid_samples is the number of validation samples.

    t_valid : array-like of shape (n_valid_samples,)
        Target vector relative to X_valid.

    d_valid : array-like of shape (n_valid_samples,)
        Vector of decision-makers/agents relative to X_valid.

    X_test : array-like of shape (n_test_samples, n_features)
        Test vector, where n_test_samples is the number of test samples.

    t_test : array-like of shape (n_test_samples,)
        Target vector relative to X_test.

    d_test : array-like of shape (n_test_samples,)
        Vector of decision-makers/agents relative to X_test.

    n_prov : int
        The number of unique agents/decision-makers.

    outcome_model : BaseEstimator
        A predictive model with the standard .predict_proba(X) method.

    filename : str
        The path to which results should be written.

    plot_true_region
        Function that plots the location of the true region of disagreement.
        See methods in data_loader.py

    plot_filename : str
        Location of a plot of the resulting region. The plot shows all test points,
        which ones are in the region, as well as where the true region is.

    true_region_func
        Either a function or a list of two functions. If a function, returns 1 if 
        input is in the region of disagreement and 0 otherwise. If a list of functions,
        each function describes a region of disagreement.

    true_provider_split : array-like of shape (n_prov,)
        An array showing the groups of each agent.

    betas : list of floats
        Potential sizes of the desired region of disagreement.

    n_iter : int
        Number of iterations to run IterativeAlg for.
        
    region_X_feat_idxs: list of ints, default: None
        Subset of feature indices used to learn region model. If None, use all features.

    Returns
    -------
    result_dict : dictionary
        A results dictionary. See the documentation for get_and_save_results().
    '''

    assert secondary_class in {'LogisticRegression', 'DecisionTree', 'RandomForest', 'HyperboxProxy', 'HyperboxProxyMSL20'}
    if region_X_feat_idxs is None:
        region_X_feat_idxs = range(X_train.shape[1])
    else:
        assert np.min(region_X_feat_idxs) >= 0
        assert np.max(region_X_feat_idxs) < X_train.shape[1]
        assert len(np.unique(region_X_feat_idxs)) == len(region_X_feat_idxs)
    
    max_zscore = float('-inf')
    for beta in betas:
        beta_results = run_iterative_alg(secondary_class,
                                         X_train, X_valid, X_test,
                                         d_train, d_valid, d_test,
                                         t_train, t_valid, t_test,
                                         n_prov,
                                         outcome_model,
                                         filename[:-4] + '_beta{0:.2f}'.format(beta) + '.pkl', 
                                         plot_true_region, plot_filename, 
                                         true_region_func, true_provider_split, 
                                         beta, n_iter=5, region_X_feat_idxs=region_X_feat_idxs)
        beta_q_scores = np.empty(n_shuffles)
        for shuffle_idx in range(n_shuffles):
            d_train_shuffle = deepcopy(d_train)
            np.random.shuffle(d_train_shuffle)
            shuffle_results = run_iterative_alg(secondary_class,
                                         X_train, X_valid, X_test,
                                         d_train_shuffle, d_valid, d_test,
                                         t_train, t_valid, t_test,
                                         n_prov,
                                         outcome_model,
                                         None, # filename
                                         plot_true_region, plot_filename, 
                                         true_region_func, true_provider_split, 
                                         beta, n_iter=5)
            beta_q_scores[shuffle_idx] = shuffle_results['iter_q_scores'][-1]
            
        beta_zscore = (beta_results['iter_q_scores'][-1] - np.mean(beta_q_scores))/np.std(beta_q_scores)
        print('z-score for beta {0:.2f}'.format(beta) + ': {0:.4f}'.format(beta_zscore))
        if beta_zscore > max_zscore:
            max_zscore = beta_zscore
            max_results = beta_results
            max_beta = beta
    print('Best beta: {0:.2f}'.format(max_beta))
    return max_results

'''
Baseline methods: "direct", "UL", "CFA", and "TARNet" baselines.
'''

def train_and_eval_region_model(model_class, X_train, X_valid, X_test, metric_train, metric_valid, metric_test, 
                                true_region_func=None, true_provider_split=None, 
                                beta=0.1, pred_provider_split=None, train_resids_pred=None, valid_resids_pred=None, 
                                test_resids_pred=None):
    '''
    Trains a region model to predict the metric from X.
    Computes the provider split using residuals if pred_provider_split is not provided and residuals are.
    Evaluates the region model and provider split if true values are given.

    Parameters
    ----------
    model_class : str
        A string indicating the type of model used to find the region.
        
    X_train: numpy array
        Context features for training samples
        
    X_valid: numpy array
        Context features for validation samples
        
    X_test: numpy array
        Context features for testing samples
        
    metric_train: numpy array
        Scores for training samples used as labels for region model
        
    metric_valid: numpy array
        Scores for validation samples used as labels for region model
    
    metric_test: numpy array
        Scores for testing samples used as labels for region model
        
    true_region_func: function
        Takes in X and outputs whether sample is in region under known data generation process
        
    true_provider_split: numpy array
        Binary values for whether each provider is in group 0 or 1 under known data generation process
        
    beta: float
        Region size
        
    pred_provider_split: numpy array
        Binary values for whether each provider is in group 0 or 1 in learned model
    
    train_resids_pred: dictionary of numpy arrays
        For each pair of providers, treatment effect predictions from residual model for training samples.
        Only applies to U-learners.
        
    valid_resids_pred: dictionary of numpy arrays
        For each pair of providers, treatment effect predictions from residual model for validation samples.
        Only applies to U-learners.
        
    test_resids_pred: dictionary of numpy arrays
        For each pair of providers, treatment effect predictions from residual model for testing samples.
        Only applies to U-learners.
        
    Returns
    ----------
    region_dict: dictionary
        Contains region model, metrics, cut-off, indices of samples in region, provider split metrics, etc.
    '''
    print('Region model ' + model_class)
    if model_class == 'No region model':
        region_model_train_pred = metric_train
        region_model_valid_pred = metric_valid
        region_model_test_pred = metric_test
        region_model = None
    else:
        if model_class == 'LogisticRegression':
            region_model = train_linear_reg(X_train, metric_train, X_valid, metric_valid)
        elif model_class == 'DecisionTree':
            region_model = train_decision_tree_reg(X_train, metric_train, X_valid, metric_valid)
        else:
            region_model = train_random_forest_reg(X_train, metric_train, X_valid, metric_valid)
        region_model_train_pred = region_model.predict(X_train)
        region_model_valid_pred = region_model.predict(X_valid)
        region_model_test_pred = region_model.predict(X_test)
        print('Region model train MSE: {0:.4f}'.format(mean_squared_error(metric_train, region_model_train_pred)))
        print('Region model valid MSE: {0:.4f}'.format(mean_squared_error(metric_valid, region_model_valid_pred)))
        print('Region model test MSE: {0:.4f}'.format(mean_squared_error(metric_test, region_model_test_pred)))
    cutoff = compute_diff_perc_cutoff(np.concatenate((region_model_train_pred, region_model_valid_pred)), 100*(1.-beta))
    train_region_idxs = np.nonzero(np.where(region_model_train_pred >= cutoff, 1, 0))
    valid_region_idxs = np.nonzero(np.where(region_model_valid_pred >= cutoff, 1, 0))
    test_region_idxs = np.nonzero(np.where(region_model_test_pred >= cutoff, 1, 0))
    region_precision = None
    region_recall = None
    region_auc = None
    if true_region_func is not None:
        region_precision, region_recall, region_auc \
            = eval_region_precision_recall_auc(region_model_test_pred, cutoff, X_test, true_region_func)
    partition_acc = None
    test_partition_acc = None
    pred_provider_split_test = pred_provider_split
    if true_provider_split is not None:
        if pred_provider_split is None:
            # for U-learner, split depends on which samples are in region
            assert train_resids_pred is not None
            assert valid_resids_pred is not None
            assert test_resids_pred is not None
            train_valid_resids_pred_in_region = dict()
            test_resids_pred_in_region = dict()
            for pair in train_resids_pred.keys():
                train_valid_resids_pred_in_region[pair] = np.concatenate((train_resids_pred[pair][train_region_idxs], \
                                                                          valid_resids_pred[pair][valid_region_idxs]))
                test_resids_pred_in_region[pair] = test_resids_pred[pair][test_region_idxs]
            pred_provider_split = compute_pairwise_provider_split(train_valid_resids_pred_in_region)
            pred_provider_split_test = compute_pairwise_provider_split(test_resids_pred_in_region)
        partition_acc = eval_provider_split_acc(pred_provider_split, true_provider_split)
        if pred_provider_split_test is not None:
            test_partition_acc = eval_provider_split_acc(pred_provider_split_test, true_provider_split)
        else:
            test_partition_acc = partition_acc
    region_dict = {'region_model': region_model, 'region_precision': region_precision, 'region_recall': region_recall, \
                   'region_auc': region_auc, 'partition_acc': partition_acc, 'cutoff': cutoff, \
                   'pred_provider_split': pred_provider_split, 'train_region_idxs': train_region_idxs, \
                   'valid_region_idxs': valid_region_idxs, 'test_region_idxs': test_region_idxs, \
                   'pred_provider_split_test': pred_provider_split_test, 'test_partition_acc': test_partition_acc}
    return region_dict

def run_model_direct(X, d, t, train_idxs, valid_idxs, test_idxs, filename, true_region_func=None, \
                     true_provider_split=None, beta=0.1):
    '''
    Runs the direct baseline.
    Trains t = f(x) and t = g(x, d).
    Region difference is then |t - f(x)| - |t  - g(x, d)|

    Parameters
    ----------
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
        
    train_idxs: list of ints
        Indices in training set
        
    valid_idxs: list of ints
        Indices in validation set
        
    test_idxs: list of ints
        Indices in test set
        
    filename: str
        Location for results pickle file
        
    true_region_func: function
        Takes in X and outputs whether sample is in region under known data generation process
        
    true_provider_split: numpy array
        Binary values for whether each provider is in group 0 or 1 under known data generation process
        
    beta: float
        Region size
        
    Returns
    -------
    result_dict : dictionary
        Contains pair of models with and without agent and metrics for region.
    '''
    n_prov = len(np.unique(d))
    assert np.min(d) == 0
    assert np.max(d) == n_prov - 1
    if n_prov == 2:
        d_one_hot = d.reshape((len(d), 1))
    else:
        d_one_hot = np.zeros((d.shape[0], n_prov))
        for i in range(n_prov):
            d_one_hot[:,i] = np.where(d == i, 1, 0)
        d_one_hot = d_one_hot[:,:-1] # To avoid collinearity
    X_train = X[train_idxs]
    d_train = d[train_idxs]
    assert len(np.unique(d_train)) == n_prov
    d_train_one_hot = d_one_hot[train_idxs]
    t_train = t[train_idxs]
    X_valid = X[valid_idxs]
    d_valid = d[valid_idxs]
    d_valid_one_hot = d_one_hot[valid_idxs]
    t_valid = t[valid_idxs]
    X_test = X[test_idxs]
    d_test = d[test_idxs]
    d_test_one_hot = d_one_hot[test_idxs]
    t_test = t[test_idxs]
    scaler = StandardScaler()
    d_train_valid_one_hot = np.vstack((d_train_one_hot, d_valid_one_hot))
    scaler.fit(d_train_valid_one_hot)
    Xd_train = np.hstack((X_train, scaler.transform(d_train_one_hot)))
    Xd_valid = np.hstack((X_valid, scaler.transform(d_valid_one_hot)))
    Xd_test = np.hstack((X_test, scaler.transform(d_test_one_hot)))
    # learn models t = f(x) and t = g(x, d)
    outcome_model = train_logreg(X_train, t_train, X_valid, t_valid)
    Xdt_model = train_logreg(Xd_train, t_train, Xd_valid, t_valid)
    Xt_train_pred = outcome_model.predict_proba(X_train)[:,1]
    Xdt_train_pred = Xdt_model.predict_proba(Xd_train)[:,1]
    Xt_valid_pred = outcome_model.predict_proba(X_valid)[:,1]
    Xdt_valid_pred = Xdt_model.predict_proba(Xd_valid)[:,1]
    Xt_test_pred = outcome_model.predict_proba(X_test)[:,1]
    Xdt_test_pred = Xdt_model.predict_proba(Xd_test)[:,1]
    print('Train Y = f(X) AUC: {0:.4f}'.format(roc_auc_score(t_train, Xt_train_pred)))
    print('Train Y = g(X, A) AUC: {0:.4f}'.format(roc_auc_score(t_train, Xdt_train_pred)))
    print('Valid Y = f(X) AUC: {0:.4f}'.format(roc_auc_score(t_valid, Xt_valid_pred)))
    print('Valid Y = g(X, A) AUC: {0:.4f}'.format(roc_auc_score(t_valid, Xdt_valid_pred)))
    print('Test Y = f(X) AUC: {0:.4f}'.format(roc_auc_score(t_test, Xt_test_pred)))
    print('Test Y = g(X, A) AUC: {0:.4f}'.format(roc_auc_score(t_test, Xdt_test_pred)))
    # Compute |t - f(x)| - |t  - g(x, d)|
    train_error_diff = compute_error_diff(t_train, Xt_train_pred, Xdt_train_pred)
    valid_error_diff = compute_error_diff(t_valid, Xt_valid_pred, Xdt_valid_pred)
    test_error_diff = compute_error_diff(t_test, Xt_test_pred, Xdt_test_pred)
    pred_provider_split = compute_logistic_provider_split(Xdt_model['logReg'].coef_.flatten()[X.shape[1]:])
    results_dict = {'X': X, 'd': d, 't': t, 'train_idxs': train_idxs, 'valid_idxs': valid_idxs, 'test_idxs': test_idxs, \
                    'outcome_model': outcome_model, 'Xdt_model': Xdt_model}
    for model_class in ['No region model', 'LogisticRegression', 'DecisionTree', 'RandomForest']:
        results_dict[model_class] \
            = train_and_eval_region_model(model_class, X_train, X_valid, X_test, train_error_diff, valid_error_diff, \
                                          test_error_diff, true_region_func, true_provider_split, beta, pred_provider_split)
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict

def train_causal_forest(X_train, d_train, t_train, X_valid, oracle_preds_valid):
    '''
    Train a causal forest
    
    Parameters
    ----------
    X_train: numpy array
        Contains context of training samples
        
    d_train: numpy array
        Contains agents of training samples. Binary.
        
    t_train: numpy array
        Contains treatment decisions of training samples
        
    X_valid: numpy array
        Contains context of validation samples
        
    oracle_preds_valid: numpy array
        Contains oracle effects of different agents for each validation sample. Used for tuning hyperparameters.
        
    Returns
    -------
    forest: CausalForest object
    '''
    # learns a causal forest, called by run_causal_forest_efficient baseline
    assert len(d_train.shape) == 1
    d_train = d_train.reshape((len(d_train),1))
    best_valid_mse = float('inf')
    for n_trees, min_leaf_size in product([12, 24, 100], [10, 25, 100]):
        forest = CausalForest(n_estimators=n_trees, min_samples_leaf=min_leaf_size, random_state=0)
        forest.fit(X=X_train, T=d_train, y=t_train)
        valid_preds = forest.predict(X_valid).flatten()
        valid_mse = np.sum(np.square(oracle_preds_valid - valid_preds))
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_forest = forest
    return forest

def compute_total_diff(preds_dict):
    '''
    Computes variation across all providers for each sample
    
    Parameters
    ----------
    preds_dict: dictionary of numpy arrays
        Maps each pair of providers to the differences in decisions for each pair
        
    Returns
    -------
    total_preds_diff: numpy array
        Sum of absolute value of differences between each pair of providers for each sample
    '''
    total_preds_diff = np.zeros(preds_dict[list(preds_dict.keys())[0]].shape)
    for pair in preds_dict.keys():
        total_preds_diff += np.abs(preds_dict[pair])
    return total_preds_diff

def compute_pairwise_provider_split(preds_dict):
    '''
    Greedily split providers into two groups to maximize difference between outcomes of two groups
    
    Parameters
    ----------
    preds_dict: dictionary of numpy arrays
        Maps each pair of providers to the differences in decisions for each pair
        
    Returns
    -------
    partition_split: numpy array
        Binary array for whether each provider is in group 0 or 1
    '''
    # set most different pair to opposite sides
    # other providers are added to whichever provider in original pair they are closer to
    max_pair_total_diff = 0
    max_prov = 1
    for pair in preds_dict.keys():
        pair_total_diff = np.sum(np.abs(preds_dict[pair]))
        if pair_total_diff > max_pair_total_diff:
            max_pair = pair
            max_pair_total_diff = pair_total_diff
        pair_prov1 = int(pair[pair.index(',')+1:pair.index(')')])
        if pair_prov1 > max_prov:
            max_prov = pair_prov1
    if max_pair_total_diff == 0:
        print('Providers all same. Random partition.')
        return np.random.randint(2, size=max_prov+1) # random split if providers seem all the same
    part0_prov = int(max_pair[1:max_pair.index(',')])
    part1_prov = int(max_pair[max_pair.index(',')+1:max_pair.index(')')])
    partition_split = np.zeros(max_prov+1)
    for i in range(len(partition_split)):
        if i == part0_prov or i == part1_prov:
            continue
        if i > part0_prov:
            part0_pair_str = '(' + str(part0_prov) + ',' + str(i) + ')'
        else:
            part0_pair_str = '(' + str(i) + ',' + str(part0_prov) + ')'
        if i > part1_prov:
            part1_pair_str = '(' + str(part1_prov) + ',' + str(i) + ')'
        else:
            part1_pair_str = '(' + str(i) + ',' + str(part1_prov) + ')'
        if np.sum(np.abs(preds_dict[part0_pair_str])) > np.sum(np.abs(preds_dict[part1_pair_str])):
            partition_split[i] = 1
    return partition_split

def run_causal_forest_efficient(X, d, t, train_idxs, valid_idxs, test_idxs, oracle_preds, filename, true_region_func=None, \
                                true_provider_split=None, beta=.1):
    '''
    Runs causal forest adaptation (CFA)
    
    Parameters
    ----------
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
        
    train_idxs: list of ints
        Indices in training set
        
    valid_idxs: list of ints
        Indices in validation set
        
    test_idxs: list of ints
        Indices in validation set
        
    oracle_preds: dictionary of treatment effect differences between each pair of agents
        Used to initialize CFA and tune causal forests
        Could be generated using U-learners
        
    filename: str
        Location for results pickle file
        
    true_region_func: function
        Takes in X and outputs whether sample is in region under known data generation process
        
    true_provider_split: numpy array
        Binary values for whether each provider is in group 0 or 1 under known data generation process
        
    beta: float
        Region size
        
    Returns
    -------
    result_dict : dictionary
        Contains models and metrics for region.
    '''
    '''
    Only compute O(P) causal forests, where P is the number of providers.
    Heuristic:
    1. Initialize the grouping using oracle_preds (train + valid):
        a. Start with the pair (a,b) that has the largest absolute difference in oracle_preds.
        b. For any provider c, if ||c - a|| < .1 * ||c - b||, add c to a's group A. Same for adding c to b's group B.
    2. Add the remaining providers to the groups using causal forests:
        a. Order the providers c by sum_d in A, B ||c - d|| -> small quantity should be smaller change when add to grouping
        b. In that order, for each provider c:
            b1. Compute CF(T0 = A + c, T1 = B) and CF(T0 = A, T1 = B + c).
            b2. Whichever causal forest computes larger absolute treatment effect across all samples, add c to that group.
    3. Return the grouping from above and the region with the largest absolute treatment effect.
    '''
    n_prov = len(np.unique(d))
    assert np.min(d) == 0
    assert np.max(d) == n_prov - 1
    # split data
    X_train = X[train_idxs]
    d_train = d[train_idxs]
    assert len(np.unique(d_train)) == n_prov
    t_train = t[train_idxs]
    X_valid = X[valid_idxs]
    d_valid = d[valid_idxs]
    t_valid = t[valid_idxs]
    X_test = X[test_idxs]
    d_test = d[test_idxs]
    t_test = t[test_idxs]
    train_valid_idxs = np.concatenate((train_idxs, valid_idxs))
    # Step 1a
    pair_total_diff_dict = dict()
    max_pair_total_diff = 0 
    for pair in oracle_preds.keys():
        pair_total_diff = np.sum(np.abs(oracle_preds[pair]))
        pair_total_diff_dict[pair] = pair_total_diff
        if pair_total_diff > max_pair_total_diff:
            max_pair_total_diff = pair_total_diff
            max_pair = pair
    if max_pair_total_diff == 0:
        return 'oracle_preds all 0'
    max_pair_prov0 = int(max_pair[1:max_pair.index(',')])
    max_pair_prov1 = int(max_pair[max_pair.index(',')+1:-1])
    group0_provs = set([max_pair_prov0])
    group1_provs = set([max_pair_prov1])
    provs_to_add = []
    provs_to_add_total_diff = []
    set_oracle_preds = oracle_preds[max_pair]
    # Step 1b
    for prov in range(n_prov):
        if prov == max_pair_prov0 or prov == max_pair_prov1:
            continue
        if prov < max_pair_prov0:
            prov_pair0 = '(' + str(prov) + ',' + str(max_pair_prov0) + ')'
        else:
            prov_pair0 = '(' + str(max_pair_prov0) + ',' + str(prov) + ')'
        if prov < max_pair_prov1:
            prov_pair1 = '(' + str(prov) + ',' + str(max_pair_prov1) + ')'
        else:
            prov_pair1 = '(' + str(max_pair_prov1) + ',' + str(prov) + ')'
        if pair_total_diff_dict[prov_pair0] < .1*pair_total_diff_dict[prov_pair1]:
            group0_provs.add(prov)
            for group1_prov in group1_provs:
                if prov < group1_prov:
                    new_pair = '(' + str(prov) + ',' + str(group1_prov) + ')'
                    set_oracle_preds += oracle_preds[new_pair]
                else:
                    new_pair = '(' + str(group1_prov) + ',' + str(prov) + ')'
                    set_oracle_preds -= oracle_preds[new_pair]
        elif pair_total_diff_dict[prov_pair1] < .1*pair_total_diff_dict[prov_pair0]:
            group1_provs.add(prov)
            for group0_prov in group0_provs:
                if prov < group0_prov:
                    new_pair = '(' + str(prov) + ',' + str(group0_prov) + ')'
                    set_oracle_preds -= oracle_preds[new_pair]
                else:
                    new_pair = '(' + str(group0_prov) + ',' + str(prov) + ')'
                    set_oracle_preds += oracle_preds[new_pair]
        else:
            provs_to_add.append(prov)
            provs_to_add_total_diff.append(pair_total_diff_dict[prov_pair0] + pair_total_diff_dict[prov_pair1])
    print('Initialized with ' + str(len(group0_provs)) + ' providers in group 0 and ' + str(len(group1_provs)) \
          + ' providers in group 1')
    # Step 2a
    if len(provs_to_add_total_diff) == 0:
        d_train_group = np.where(np.isin(d_train, np.array(list(group1_provs))), 1, 0)
        final_CF = train_causal_forest(X_train, d_train_group, t_train, X_valid, \
                                      set_oracle_preds[valid_idxs])
        final_CF_preds = final_CF.predict(X).flatten()
    else:
        provs_to_add_sorted_idxs = np.argsort(np.array(provs_to_add_total_diff))
        for i in provs_to_add_sorted_idxs:
            prov = provs_to_add[provs_to_add_sorted_idxs[i]]
            set_provs = np.concatenate((np.array(list(group0_provs)), np.array(list(group1_provs)), np.array([prov])))
            set_train_idxs = np.nonzero(np.where(np.isin(d_train, set_provs), 1, 0))[0]
            set_valid_idxs = np.nonzero(np.where(np.isin(d_valid, set_provs), 1, 0))[0]
            X_train_set = X_train[set_train_idxs]
            X_valid_set = X_valid[set_valid_idxs]
            d_train_set = d_train[set_train_idxs]
            t_train_set = t_train[set_train_idxs]
            d_train_set0 = np.where(np.isin(d_train_set, np.array(list(group1_provs))), 1, 0)
            d_train_set1 = np.where(np.isin(d_train_set, np.array(list(group0_provs))), 0, 1)
            # set up oracle predictions for step 2b1
            set0_oracle_preds_additions = np.zeros(set_oracle_preds.shape)
            for group1_prov in group1_provs:
                if prov < group1_prov:
                    new_pair = '(' + str(prov) + ',' + str(group1_prov) + ')'
                    set0_oracle_preds_additions += oracle_preds[new_pair]
                else:
                    new_pair = '(' + str(group1_prov) + ',' + str(prov) + ')'
                    set0_oracle_preds_additions -= oracle_preds[new_pair]
            set1_oracle_preds_additions = np.zeros(set_oracle_preds.shape)
            for group0_prov in group0_provs:
                if prov < group0_prov:
                    new_pair = '(' + str(prov) + ',' + str(group0_prov) + ')'
                    set1_oracle_preds_additions -= oracle_preds[new_pair]
                else:
                    new_pair = '(' + str(group0_prov) + ',' + str(prov) + ')'
                    set1_oracle_preds_additions += oracle_preds[new_pair]
            set0_oracle_preds = set_oracle_preds + set0_oracle_preds_additions
            set1_oracle_preds = set_oracle_preds + set1_oracle_preds_additions
            # Step 2b1
            set0_CF = train_causal_forest(X_train_set, d_train_set0, t_train_set, X_valid_set, \
                                          set0_oracle_preds[valid_idxs][set_valid_idxs])
            set1_CF = train_causal_forest(X_train_set, d_train_set1, t_train_set, X_valid_set, \
                                          set1_oracle_preds[valid_idxs][set_valid_idxs])
            set0_CF_preds = set0_CF.predict(X).flatten()
            set1_CF_preds = set1_CF.predict(X).flatten()
            # Step 2b2
            if np.sum(np.abs(set0_CF_preds[train_valid_idxs])) > np.sum(np.abs(set1_CF_preds[train_valid_idxs])):
                group0_provs.add(prov)
                set_oracle_preds = set0_oracle_preds
                if i == len(provs_to_add_sorted_idxs) - 1:
                    final_CF = set0_CF
                    final_CF_preds = set0_CF_preds
            else:
                group1_provs.add(prov)
                set_oracle_preds = set1_oracle_preds
                if i == len(provs_to_add_sorted_idxs) - 1:
                    final_CF = set1_CF
                    final_CF_preds = set1_CF_preds
    # Step 3
    results_dict = {'X': X, 'd': d, 't': t, 'train_idxs': train_idxs, 'valid_idxs': valid_idxs, 'test_idxs': test_idxs, \
                    'preds': final_CF_preds}
    pred_provider_split = np.zeros(n_prov)
    for prov in group1_provs:
        pred_provider_split[prov] = 1
    for model_class in ['No region model', 'LogisticRegression', 'DecisionTree', 'RandomForest']:
        results_dict[model_class] \
            = train_and_eval_region_model(model_class, X_train, X_valid, X_test, np.abs(final_CF_preds[train_idxs]), \
                                          np.abs(final_CF_preds[valid_idxs]), np.abs(final_CF_preds[test_idxs]), \
                                          true_region_func, true_provider_split, beta, pred_provider_split)
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict

def run_U_learner(X, d, t, train_idxs, valid_idxs, test_idxs, filename, true_region_func=None, \
                  true_provider_split=None, beta=.1):
    '''
    Runs U-learner method
    
    Parameters
    ----------
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
        
    train_idxs: list of ints
        Indices in training set
        
    valid_idxs: list of ints
        Indices in validation set
        
    test_idxs: list of ints
        Indices in validation set
        
    filename: str
        Location for results pickle file
        
    true_region_func: function
        Takes in X and outputs whether sample is in region under known data generation process
        
    true_provider_split: numpy array
        Binary values for whether each provider is in group 0 or 1 under known data generation process
        
    beta: float
        Region size
        
    Returns
    -------
    result_dict : dictionary
        Contains models and metrics for region.
    '''
    n_prov = len(np.unique(d))
    assert np.min(d) == 0
    assert np.max(d) == n_prov - 1
    # split data
    X_train = X[train_idxs]
    d_train = d[train_idxs]
    assert len(np.unique(d_train)) == n_prov
    t_train = t[train_idxs]
    X_valid = X[valid_idxs]
    d_valid = d[valid_idxs]
    t_valid = t[valid_idxs]
    X_test = X[test_idxs]
    d_test = d[test_idxs]
    t_test = t[test_idxs]
    # create data dictionary containing (X_train, d_train, X_valid, d_valid) for every pair of providers
    pair_data_dict = dict()
    if n_prov == 2:
        pair_data_dict['(0,1)'] = {'X_train': X_train, 'd_train': d_train, 'X_valid': X_valid, 'd_valid': d_valid, \
                                   'X_test': X_test, 'd_test': d_test, \
                                   'train_pair_idxs': np.arange(X_train.shape[0]), \
                                   'valid_pair_idxs': np.arange(X_valid.shape[0]), \
                                   'test_pair_idxs': np.arange(X_test.shape[0])}
    else:
        for i in range(n_prov):
            for j in range(i+1, n_prov):
                pair_str = '(' + str(i) + ',' + str(j) + ')'
                train_pair_idxs = np.nonzero(np.where(np.logical_or(d_train==i, d_train==j), 1, 0))[0]
                d_train_pair = np.where(d_train[train_pair_idxs]==j, 1, 0)
                assert len(np.unique(d_train_pair)) == 2
                X_train_pair = X_train[train_pair_idxs]
                valid_pair_idxs = np.nonzero(np.where(np.logical_or(d_valid==i, d_valid==j), 1, 0))[0]
                d_valid_pair = np.where(d_valid[valid_pair_idxs]==j, 1, 0)
                assert len(np.unique(d_valid_pair)) == 2
                X_valid_pair = X_valid[valid_pair_idxs]
                test_pair_idxs = np.nonzero(np.where(np.logical_or(d_test==i, d_test==j), 1, 0))[0]
                d_test_pair = np.where(d_test[test_pair_idxs]==j, 1, 0)
                assert len(np.unique(d_test_pair)) == 2
                X_test_pair = X_test[test_pair_idxs]
                pair_data_dict[pair_str] = {'X_train': X_train_pair, 'd_train': d_train_pair, 'X_valid': X_valid_pair, \
                                            'd_valid': d_valid_pair, 'X_test': X_test_pair, 'd_test': d_test_pair, \
                                            'train_pair_idxs': train_pair_idxs, 'valid_pair_idxs': valid_pair_idxs, \
                                            'test_pair_idxs': test_pair_idxs}
    # learn d = f(X) for every pair of providers and t = g(X) from all the data
    model_d_from_X = dict()
    for pair in pair_data_dict.keys():
        model_d_from_X[pair] = train_logreg(pair_data_dict[pair]['X_train'], pair_data_dict[pair]['d_train'], \
                                            pair_data_dict[pair]['X_valid'], pair_data_dict[pair]['d_valid'])
    model_t_from_X = train_logreg(X_train, t_train, X_valid, t_valid)
    # predict t and d using models and compute residual (t - g(X))/(d - f(X)) for each pair of providers
    train_t_preds = model_t_from_X.predict_proba(X_train)[:,1]
    valid_t_preds = model_t_from_X.predict_proba(X_valid)[:,1]
    test_t_preds = model_t_from_X.predict_proba(X_test)[:,1]
    print('Train y = g(x) AUC: {0:.4f}'.format(roc_auc_score(t_train, train_t_preds)))
    print('Valid y = g(x) AUC: {0:.4f}'.format(roc_auc_score(t_valid, valid_t_preds)))
    print('Test y = g(x) AUC: {0:.4f}'.format(roc_auc_score(t_test, test_t_preds)))
    train_d_preds = dict()
    valid_d_preds = dict()
    test_d_preds = dict()
    train_resid_ratios = dict()
    valid_resid_ratios = dict()
    test_resid_ratios = dict()
    for pair in model_d_from_X.keys():
        print(pair)
        train_d_preds[pair] = model_d_from_X[pair].predict_proba(pair_data_dict[pair]['X_train'])[:,1]
        valid_d_preds[pair] = model_d_from_X[pair].predict_proba(pair_data_dict[pair]['X_valid'])[:,1]
        test_d_preds[pair] = model_d_from_X[pair].predict_proba(pair_data_dict[pair]['X_test'])[:,1]
        print('Train a = f(x) AUC: {0:.4f}'.format(roc_auc_score(pair_data_dict[pair]['d_train'], train_d_preds[pair])))
        print('Valid a = f(x) AUC: {0:.4f}'.format(roc_auc_score(pair_data_dict[pair]['d_valid'], valid_d_preds[pair])))
        print('Test a = f(x) AUC: {0:.4f}'.format(roc_auc_score(pair_data_dict[pair]['d_test'], test_d_preds[pair])))
        train_d_resid = pair_data_dict[pair]['d_train'] - train_d_preds[pair]
        clipped_train_d_resid = np.where(np.logical_and(train_d_resid >= 0, train_d_resid < 1e-5), 1e-5, train_d_resid)
        clipped_train_d_resid = np.where(np.logical_and(clipped_train_d_resid > -1e-5, clipped_train_d_resid < 0), \
                                         -1e-5, clipped_train_d_resid)
        valid_d_resid = pair_data_dict[pair]['d_valid'] - valid_d_preds[pair]
        clipped_valid_d_resid = np.where(np.logical_and(valid_d_resid >= 0, valid_d_resid < 1e-5), 1e-5, valid_d_resid)
        clipped_valid_d_resid = np.where(np.logical_and(clipped_valid_d_resid > -1e-5, clipped_valid_d_resid < 0), \
                                         -1e-5, clipped_valid_d_resid)
        test_d_resid = pair_data_dict[pair]['d_test'] - test_d_preds[pair]
        clipped_test_d_resid = np.where(np.logical_and(test_d_resid >= 0, test_d_resid < 1e-5), 1e-5, test_d_resid)
        clipped_test_d_resid = np.where(np.logical_and(clipped_test_d_resid > -1e-5, clipped_test_d_resid < 0), \
                                        -1e-5, clipped_test_d_resid)
        train_resid_ratios[pair] = (t_train[pair_data_dict[pair]['train_pair_idxs']] \
                                    - train_t_preds[pair_data_dict[pair]['train_pair_idxs']])/clipped_train_d_resid
        valid_resid_ratios[pair] = (t_valid[pair_data_dict[pair]['valid_pair_idxs']] \
                                    - valid_t_preds[pair_data_dict[pair]['valid_pair_idxs']])/clipped_valid_d_resid
        test_resid_ratios[pair] = (t_test[pair_data_dict[pair]['test_pair_idxs']] \
                                   - test_t_preds[pair_data_dict[pair]['test_pair_idxs']])/clipped_test_d_resid
    # learn residual = h(X) model for each pair of providers
    model_resids = dict()
    for pair in train_resid_ratios.keys():
        model_resids[pair] = train_linear_reg(pair_data_dict[pair]['X_train'], train_resid_ratios[pair], \
                                              pair_data_dict[pair]['X_valid'], valid_resid_ratios[pair])
        train_resids_pred_pair = model_resids[pair].predict(pair_data_dict[pair]['X_train'])
        valid_resids_pred_pair = model_resids[pair].predict(pair_data_dict[pair]['X_valid'])
        test_resids_pred_pair = model_resids[pair].predict(pair_data_dict[pair]['X_test'])
        print(pair)
        print('Train resid h(x) MSE: {0:.4f}'.format(mean_squared_error(train_resid_ratios[pair], train_resids_pred_pair)))
        print('Valid resid h(x) MSE: {0:.4f}'.format(mean_squared_error(valid_resid_ratios[pair], valid_resids_pred_pair)))
        print('Test resid h(x) MSE: {0:.4f}'.format(mean_squared_error(test_resid_ratios[pair], test_resids_pred_pair)))
    # predict residual for all samples using h(X) for each pair of providers
    train_resids_pred = dict()
    valid_resids_pred = dict()
    test_resids_pred = dict()
    all_resids_pred = dict()
    for pair in train_resid_ratios.keys():
        train_resids_pred[pair] = model_resids[pair].predict(X_train)
        valid_resids_pred[pair] = model_resids[pair].predict(X_valid)
        test_resids_pred[pair] = model_resids[pair].predict(X_test)
        all_resids_pred[pair] = model_resids[pair].predict(X)
    # compute total difference across all pairs of providers for each sample
    train_total_diff = compute_total_diff(train_resids_pred)
    valid_total_diff = compute_total_diff(valid_resids_pred)
    test_total_diff = compute_total_diff(test_resids_pred)
    # save data to return
    results_dict = {'X': X, 'd': d, 't': t, 'train_idxs': train_idxs, 'valid_idxs': valid_idxs, 'test_idxs': test_idxs, \
                    'pair_data_dict': pair_data_dict, 'model_d_from_X': model_d_from_X, 'model_t_from_X': model_t_from_X, \
                    'train_resids_pred': train_resids_pred, 'valid_resids_pred': valid_resids_pred, \
                    'test_resids_pred': test_resids_pred, 'all_resids_pred': all_resids_pred, 'model_resids': model_resids}
    for model_class in ['No region model', 'LogisticRegression', 'DecisionTree', 'RandomForest']:
        results_dict[model_class] \
            = train_and_eval_region_model(model_class, X_train, X_valid, X_test, np.abs(train_total_diff), \
                                          np.abs(valid_total_diff), np.abs(test_total_diff), \
                                          true_region_func, true_provider_split, beta, None, train_resids_pred, \
                                          valid_resids_pred, test_resids_pred)
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict

class TARNet(nn.Module):
    def __init__(self, num_feats, num_agents, num_units, dropout, nonlin=nn.ReLU()):
        '''
        Initializes a TARNet model
        num_feats: int
            Number of input features
            
        num_agents: int
            Number of prediction heads
            
        num_units: int
            Number of units in 2 hidden layers
            
        dropout: float
            Dropout rate
            
        nonlin: function
            Non-linearity after each layer
        '''
        super(TarNet, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(num_feats, num_units),
            nonlin,
            nn.Dropout(dropout),
            nn.Linear(num_units, num_units),
            nonlin,
            nn.Dropout(dropout)
        )
        
        # Variable number of independent heads, one per agent
        self.heads = nn.ModuleList([])
        for _ in range(num_agents):
            self.heads.append(nn.Linear(num_units, 1))

    def forward(self, X, a):
        '''
        Outputs a list of predictions of size (n_samples, n_classes)
        where each element in the list corresponds to a different head
        
        Parameters
        ----------
        X: torch float tensor
            features of samples
            
        a: torch float tensor
            agents of samples, determines which prediction head to use
        '''
        common_features = self.shared(X)
        outputs = []
        for head in self.heads:
            outputs.append(head(common_features))
            
        all_out = torch.cat(outputs, dim=1)
        
        # returns a particular head; For debugging, you can run 
        #   return all_out        
        # and then check the heads yourself!
        return all_out[a]

def plot_epoch_losses(train_losses, valid_losses, plot_title, plot_filename):
    '''
    Plots the training and validation losses across epochs
    
    Parameters
    ----------
    train_losses: list of floats
        Losses across epochs
        
    valid_losses: list of floats
        Losses across epochs
        
    plot_title: str
        Title of plot
        
    plot_filename: str
        Location to save plot
    '''
    plt.clf()
    plt.plot(train_losses, label='train', linestyle='--')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(plot_filename)
    
def run_tarnet(X, d, t, train_idxs, valid_idxs, test_idxs, filename, output_dir, true_region_func=None, \
               true_provider_split=None, beta=.1):
    '''
    Runs the TARNet baseline, which learns a shared representation and heads for each agent

    Parameters
    ----------
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
        
    train_idxs: list of ints
        Indices in training set
        
    valid_idxs: list of ints
        Indices in validation set
        
    test_idxs: list of ints
        Indices in validation set
        
    filename: str
        Location for results pickle file
        
    output_dir: str
        Directory for saved models and results
        
    true_region_func: function
        Takes in X and outputs whether sample is in region under known data generation process
        
    true_provider_split: numpy array
        Binary values for whether each provider is in group 0 or 1 under known data generation process
        
    beta: float
        Region size
        
    Returns
    -------
    result_dict : dictionary
        Contains best hyperparameters of model and metrics for region.
    '''
    n_prov = len(np.unique(d))
    assert np.min(d) == 0
    assert np.max(d) == n_prov - 1
    d_one_hot = np.zeros((d.shape[0], n_prov))
    for i in range(n_prov):
        d_one_hot[:,i] = np.where(d == i, 1, 0)
    d_one_hot = d_one_hot.astype(bool)
    X_train = X[train_idxs]
    d_train_one_hot = d_one_hot[train_idxs]
    assert np.all(np.sum(d_train_one_hot, axis=0) > 0)
    d_train_one_hot = torch.BoolTensor(d_train_one_hot).cuda()
    t_train = torch.FloatTensor(t[train_idxs]).cuda()
    X_valid = X[valid_idxs]
    d_valid_one_hot = torch.BoolTensor(d_one_hot[valid_idxs]).cuda()
    t_valid = torch.FloatTensor(t[valid_idxs]).cuda()
    X_test = X[test_idxs]
    d_test_one_hot = torch.BoolTensor(d_one_hot[test_idxs]).cuda()
    t_test = torch.FloatTensor(t[test_idxs]).cuda()
    scaler = StandardScaler()
    scaler.fit(np.concatenate((X_train, X_valid)))
    X_train_scaled = torch.FloatTensor(scaler.transform(X_train)).cuda()
    X_valid_scaled = torch.FloatTensor(scaler.transform(X_valid)).cuda()
    X_test_scaled = torch.FloatTensor(scaler.transform(X_test)).cuda()
    
    best_models = []
    best_model_valid_losses = []
    lr_header = {1e-4: '4', 5e-4: '0005', 1e-3: '3', 5e-3: '005', 1e-2: '2'}
    momentum_header = {0: '0'}
    dropout_header = {0.05: '05', 0.25: '25'}
    n_epochs = 200
        
    for lr, momentum, num_units, dropout in product([1e-4, 1e-3, 1e-2], [0], [10, 20], [0.05, 0.25]):
        model = TARNet(num_feats=X_train_scaled.shape[1], num_agents=n_prov, num_units=num_units, dropout=dropout).cuda()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
        # loss below includes sigmoid function, sum for backprop, mean for evaluation
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        loss_fn_eval = nn.BCEWithLogitsLoss(reduction='mean')
        train_losses = []
        valid_losses = []
        setting_folder = 'tarnet_lr' + lr_header[lr] + '_momentum' + momentum_header[momentum] + '_nunits' + str(num_units) \
            + '_dropout' + dropout_header[dropout] + '/'
        if not os.path.exists(output_dir + setting_folder):
            os.makedirs(output_dir + setting_folder)
        
        for epoch in range(n_epochs):
            model.train()
            train_agent_out = model(X_train_scaled, d_train_one_hot)
            train_loss = loss_fn(train_agent_out, t_train)
            
            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.eval()
                train_agent_out = model(X_train_scaled, d_train_one_hot)
                train_loss = loss_fn_eval(train_agent_out, t_train)
                valid_agent_out = model(X_valid_scaled, d_valid_one_hot)
                valid_loss = loss_fn_eval(valid_agent_out, t_valid)

                train_losses.append(float(train_loss.cpu().detach().numpy()))
                valid_losses.append(float(valid_loss.cpu().detach().numpy()))

                torch.save(model.state_dict(), output_dir + setting_folder + 'epoch' + str(epoch))
        
        plot_epoch_losses(train_losses, valid_losses, 'TARNet', output_dir + setting_folder[:-1] + '_losses.pdf')
        best_epoch = np.argmin(np.array(valid_losses))
        best_model_valid_losses.append(valid_losses[best_epoch])
        best_models.append((lr, momentum, num_units, dropout, best_epoch))
        for epoch in range(n_epochs):
            if epoch != best_epoch:
                os.remove(output_dir + setting_folder + 'epoch' + str(epoch))
        print(setting_folder + 'epoch' + str(best_epoch))
        print('Train loss: {0:.4f}'.format(train_losses[best_epoch]))
        print('Valid loss: {0:.4f}'.format(valid_losses[best_epoch]))
        
    model_idx = np.argmin(np.array(best_model_valid_losses))
    best_lr, best_momentum, best_num_units, best_dropout, best_epoch = best_models[model_idx]
    model = TarNet(num_feats=X_train_scaled.shape[1], num_agents=n_prov, num_units=best_num_units, dropout=best_dropout).cuda()
    best_model_file = 'tarnet_lr' + lr_header[best_lr] + '_momentum' + momentum_header[best_momentum] + '_nunits' \
        + str(best_num_units) + '_dropout' + dropout_header[best_dropout] + '/epoch' + str(best_epoch)
    print(best_model_file)
    model.load_state_dict(torch.load(output_dir + best_model_file))
    with torch.no_grad():
        model.eval()
        train_agent_out = model(X_train_scaled, d_train_one_hot)
        train_loss = loss_fn_eval(train_agent_out, t_train)
        valid_agent_out = model(X_valid_scaled, d_valid_one_hot)
        valid_loss = loss_fn_eval(valid_agent_out, t_valid)
        test_agent_out = model(X_test_scaled, d_test_one_hot)
        test_loss = loss_fn_eval(test_agent_out, t_test)
    print('TARNet train loss: {0:.4f}'.format(train_loss))
    print('TARNet valid loss: {0:.4f}'.format(valid_loss))
    print('TARNet test loss: {0:.4f}'.format(test_loss))
    
    # Compute variance as quantity to predict with region model
    train_all_out = model(X_train_scaled, 
                          torch.BoolTensor(np.ones(d_train_one_hot.shape).astype(bool))).reshape(d_train_one_hot.shape)
    train_variance = train_all_out.cpu().detach().numpy().var(axis=1)
    valid_all_out = model(X_valid_scaled, 
                          torch.BoolTensor(np.ones(d_valid_one_hot.shape).astype(bool))).reshape(d_valid_one_hot.shape)
    valid_variance = valid_all_out.cpu().detach().numpy().var(axis=1)
    test_all_out = model(X_test_scaled, 
                         torch.BoolTensor(np.ones(d_test_one_hot.shape).astype(bool))).reshape(d_test_one_hot.shape)
    test_variance = test_all_out.cpu().detach().numpy().var(axis=1)
    
    # save data to return
    results_dict = {'X': X, 'd': d, 't': t, 'train_idxs': train_idxs, 'valid_idxs': valid_idxs, 'test_idxs': test_idxs, 
                    'best_lr': best_lr, 'best_momentum': best_momentum, 'best_num_units': best_num_units, 
                    'best_dropout': best_dropout, 'best_epoch': best_epoch, 
                    'train_variance': train_variance, 'valid_variance': valid_variance, 'test_variance': test_variance}
    for model_class in ['No region model', 'LogisticRegression', 'DecisionTree', 'RandomForest']:
        results_dict[model_class] \
            = train_and_eval_region_model(model_class, X_train, X_valid, X_test, train_variance, 
                                          valid_variance, test_variance, true_region_func, None, beta, None) # no provider split
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict