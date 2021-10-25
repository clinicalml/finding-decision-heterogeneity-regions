import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from itertools import product


def train_logreg(
        X_train, y_train, X_valid, y_valid,
        Cs=[10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]):
    '''
    Train a logistic regression by cross-validation.

    Returns
    -------
    best_logreg
        A LogisticRegression object.
    '''
    best_valid_auc = -1
    for C in Cs:
        logreg = Pipeline(
                [('scaler', preprocessing.StandardScaler()),
                 ('logReg', LogisticRegression(
                    C=C, solver='lbfgs',
                    multi_class='auto',
                    random_state=0, max_iter=1000))])

        logreg.fit(X_train, y_train)
        valid_pred = logreg.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_logreg = logreg

    # Refit with best parameters
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_logreg = best_logreg.fit(X, y)

    return best_logreg


def train_linear_reg(X_train, y_train, X_valid, y_valid):
    '''
    Train a linear regression by cross-validation.

    Returns
    -------
    best_linear_reg
        A Ridge object.
    '''
    best_valid_mse = float('inf')
    for alpha in [0.01, 0.1, 1, 10, 100]:
        linear_reg = Pipeline(
                [('scaler', preprocessing.StandardScaler()),
                 ('ridge', Ridge(
                     alpha=alpha, random_state=0, max_iter=1000))])

        linear_reg.fit(X_train, y_train)
        valid_pred = linear_reg.predict(X_valid)
        valid_mse = np.sum(np.square(y_valid - valid_pred))
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_linear_reg = linear_reg

    # Refit with best parameters
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_linear_reg = best_linear_reg.fit(X, y)

    return best_linear_reg


def train_decision_tree(X_train, y_train, X_valid, y_valid,
                        min_samples_leaf_options=[10, 25, 100]):
    '''
    Train a decision tree classifier by cross-validation.

    Returns
    -------
    best_dectree
        A DecisionTreeClassifier object.
    '''
    assert len(y_train.shape) == 1
    best_valid_auc = -1
    # min_samples_leaf_options = [10, 25, 100]
    if X_train.shape[0] < 10:
        min_samples_leaf_options = [1]
    for min_samples_leaf in min_samples_leaf_options:
        if min_samples_leaf > X_train.shape[0]:
            continue
        dectree = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf,
                random_state=0)
        dectree.fit(X_train, y_train)
        valid_pred = dectree.predict_proba(X_valid)[:, 1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_dectree = dectree

    # Refit with best parameters
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_dectree = best_dectree.fit(X, y)

    return best_dectree


def train_decision_tree_reg(
        X_train, y_train, X_valid, y_valid, 
        min_samples_leaf_options=[10, 25, 100], 
        max_depth_options=None, min_samples_leaf_default=None):
    '''
    Train a decision tree regressor by cross-validation.

    Returns
    -------
    best_dectree
        A DecisionTreeRegressor object.
    '''
    assert min_samples_leaf_options is not None or max_depth_options is not None
    best_valid_mse = float('inf')
    if min_samples_leaf_options is not None:
        if X_train.shape[0] < 10:
            min_samples_leaf_options = [1]
        for min_samples_leaf in min_samples_leaf_options:
            if min_samples_leaf > X_train.shape[0]:
                continue
            dectree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=0)
            dectree.fit(X_train, y_train)
            valid_pred = dectree.predict(X_valid)
            valid_mse = np.sum(np.square(y_valid - valid_pred))
            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                best_dectree = dectree
    else:
        for max_depth in max_depth_options:
            if min_samples_leaf_default:
                dectree = DecisionTreeRegressor(
                        max_depth=max_depth, 
                        min_samples_leaf=min_samples_leaf_default,
                        random_state=0)
            else:
                dectree = DecisionTreeRegressor(
                        max_depth=max_depth,
                        random_state=0)
            dectree.fit(X_train, y_train)
            valid_pred = dectree.predict(X_valid)
            valid_mse = np.sum(np.square(y_valid - valid_pred))
            if valid_mse < best_valid_mse:
                best_valid_mse = valid_mse
                best_dectree = dectree

    # Refit on all the data
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_dectree = best_dectree.fit(X, y)

    return best_dectree


def train_random_forest(X_train, y_train, X_valid, y_valid, min_samples_leaf_options = [10, 25, 100]):
    '''
    Train a random forest classifier by cross-validation.

    Returns
    -------
    best_forest
        A RandomForestClassifier object.
    '''
    assert len(y_train.shape) == 1
    best_valid_auc = -1
    #min_samples_leaf_options = [10, 25, 100]
    if X_train.shape[0] < 10:
        min_samples_leaf_options = [1]
    for n_trees, min_samples_leaf in product([10, 25, 100], min_samples_leaf_options):
        if min_samples_leaf > X_train.shape[0]:
            continue
        forest = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, random_state=0)
        forest.fit(X_train, y_train)
        valid_pred = forest.predict_proba(X_valid)[:,1]
        valid_auc = roc_auc_score(y_valid, valid_pred)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_forest = forest

    # Refit on all the data
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_forest = best_forest.fit(X, y)

    return best_forest


def train_random_forest_reg(X_train, y_train, X_valid, y_valid, min_samples_leaf_options = [10, 25, 100]):
    '''
    Train a random forest regressor by cross-validation.

    Returns
    -------
    best_forest
        A RandomForestRegressor object.
    '''
    best_valid_mse = float('inf')
    if X_train.shape[0] < 10:
        min_samples_leaf_options = [1]
    for n_trees, min_samples_leaf in product([10, 25, 100], min_samples_leaf_options):
        if min_samples_leaf > X_train.shape[0]:
            continue
        forest = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, random_state=0)
        forest.fit(X_train, y_train)
        valid_pred = forest.predict(X_valid)
        valid_mse = np.sum(np.square(y_valid - valid_pred))
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_forest = forest

    # Refit on all the data
    X = np.concatenate([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    best_forest = best_forest.fit(X, y)

    return best_forest


def compute_error_diff(y_true, y_pred1, y_pred2):
    return np.abs(y_pred1 - y_true) - np.abs(y_pred2 - y_true)


def compute_diff_perc_cutoff(diffs, perc=90):
    return np.percentile(diffs, perc)


def eval_region_precision_recall_auc(diffs, cutoff, X, true_region_func):
    true_diffs = true_region_func(X)
    if type(true_diffs) != list:
        true_idxs = np.nonzero(np.where(true_region_func(X),1,0))[0]
        pred_idxs = np.nonzero(np.where(diffs >= cutoff, 1, 0))[0]
        true_and_pred_idxs = np.nonzero(np.where(np.logical_and(true_region_func(X), diffs >= cutoff), 1, 0))[0]
        if len(pred_idxs) == 0:
            precision = 0
        else:
            precision = float(len(true_and_pred_idxs))/len(pred_idxs)
        recall = float(len(true_and_pred_idxs))/len(true_idxs)
        auc = roc_auc_score(true_diffs, diffs)
        return precision, recall, auc
    else:
        assert len(true_diffs) == 2
        true_idxs = np.nonzero(np.where(true_diffs[0],1,0))[0]
        pred_idxs = np.nonzero(np.where(diffs >= cutoff, 1, 0))[0]
        true_and_pred_idxs = np.nonzero(np.where(np.logical_and(true_diffs[0], diffs >= cutoff), 1, 0))[0]
        if len(pred_idxs) == 0:
            precision0 = 0
        else:
            precision0 = float(len(true_and_pred_idxs))/len(pred_idxs)
        recall0 = float(len(true_and_pred_idxs))/len(true_idxs)
        auc0 = roc_auc_score(true_diffs[0], diffs)
        
        true_idxs = np.nonzero(np.where(true_diffs[1],1,0))[0]
        pred_idxs = np.nonzero(np.where(diffs >= cutoff, 1, 0))[0]
        true_and_pred_idxs = np.nonzero(np.where(np.logical_and(true_diffs[1], diffs >= cutoff), 1, 0))[0]
        if len(pred_idxs) == 0:
            precision1 = 0
        else:
            precision1 = float(len(true_and_pred_idxs))/len(pred_idxs)
        recall1 = float(len(true_and_pred_idxs))/len(true_idxs)
        auc1 = roc_auc_score(true_diffs[1], diffs)
        
        if auc0 > auc1:
            return precision0, recall0, auc0
        else:
            return precision1, recall1, auc1


def compute_logistic_provider_split(provider_coefs):
    # for now, just do a halfway split
    provider_coefs_full = np.zeros(len(provider_coefs) + 1)
    provider_coefs_full[:-1] = provider_coefs
    split_point = np.percentile(provider_coefs_full, 50)
    provider_coefs_split = np.where(provider_coefs_full > split_point, 1, 0)
    return provider_coefs_split


def eval_provider_split_acc(pred_split, true_split):
    acc_same = np.sum(np.where(np.equal(pred_split, true_split), 1, 0))/float(len(pred_split))
    acc_opp = np.sum(np.where(np.equal(pred_split, 1 - true_split), 1, 0))/float(len(pred_split))
    return max(acc_same, acc_opp)