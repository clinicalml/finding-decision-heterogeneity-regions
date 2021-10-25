import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from data_processing.real_data_loader import (
    load_diabetes_data, 
    load_ppmi_data
)
from baselines import (
    run_U_learner, 
    run_causal_forest_efficient as run_causal_forest, 
    run_model_direct, 
    run_iterative_alg, 
    best_G, 
    get_outcome_model
)
from sklearn.tree import plot_tree, DecisionTreeRegressor
import os.path

from realdata_analysis import (
    process_diabetes_results, 
    process_ppmi_results, 
    evaluate_heldout_fold_consistency, 
    evaluate_test_region_fold_consistency, 
    evaluate_fold_consistency
)

def run_model(model_class, secondary_class,
              X, 
              d, 
              t, 
              train_idxs, valid_idxs, test_idxs, 
              filename, 
              true_region_func=None, true_provider_split=None, 
              oracle_preds=None, 
              beta=.1, n_iter=5,
              outcome_model_class=None,
              verbose=True, region_X_feat_idxs=None, feature_names=None, pdp_filename=None):
    '''
    Runs a model on input data.

    Parameters
    ----------
    model_class : str
        The algorithm to be run. One of ['LogisticRegression', 'DecisionTree', 
        'RandomForest', 'Iterative'].

    secondary_class : str
        Parameterization of the outcome model. If model_class is 'Iterative', this also parameterizes the 
        region model. One of ['LogisticRegression', 'DecisionTree', 'RandomForest', 'Oracle'].

        UPDATE: If outcome_model_class is specified, then
        secondary_class only parameterizes the region model

    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    d : array-like of shape (n_samples,)
        Decision-makers or agents.

    t : array-like of shape (n_samples,)
        Binary decisions.

    train_idxs : array-like of shape (n_train_samples,)
        Indices of the training data points.

    valid_idxs : array-like of shape (n_valid_samples,)
        Indices of the validation data points.

    test_idxs : array-like of shape (n_test_samples,)
        Indices of the test data points.

    filename : str
        Location where results should be written.

    true_region_func : function, default=None
        Either a function or a list of two functions. If a function, returns 1 if 
        input is in the region of disagreement and 0 otherwise. If a list of functions,
        each function describes a region of disagreement.

    true_provider_split : array-like of shape (n_prov,), default=None
        True provider groupings.
    
    oracle_preds : array-like, default=None
        Oracle predictions required for model_class == 'CausalForest'.

    beta : float
        Size of the desired region of disagreement.

    n_iter : int
        Number of iterations to run IterativeAlg for.
        
    outcome_model_class: str, default=None
        Parameterization of the outcome model. Overrides secondary_class.
        
    verbose: boolean, default=True
        Prints model class, secondary class, and outcome model class.
        
    region_X_feat_idxs: list of ints, default: None
        Subset of feature indices used to learn region model. If None, use all features.
        
    feature_names: list of strings, default: None
        List of feature names for x-axis of partial dependence plots for random forest outcome model.
    
    pdp_filename: str, default: None
        Location for partial dependence plot for random forest outcome model.
    
    Returns
    -------
    results_dict : dictionary
        Dictionary containing results. See baselines.get_and_save_results() for more
        information.
    '''

    if outcome_model_class is None:
        outcome_model_class = secondary_class
    if region_X_feat_idxs is None:
        region_X_feat_idxs = range(X.shape[1])
    else:
        assert np.min(region_X_feat_idxs) >= 0
        assert np.max(region_X_feat_idxs) < X.shape[1]
        assert len(np.unique(region_X_feat_idxs)) == len(region_X_feat_idxs)
    
    if verbose:
        print(model_class, secondary_class, outcome_model_class)
    # Prepare train, valid, and test sets.
    X_train, X_valid, X_test = X[train_idxs], X[valid_idxs], X[test_idxs]
    d_train, d_valid, d_test = d[train_idxs], d[valid_idxs], d[test_idxs]
    t_train, t_valid, t_test = t[train_idxs], t[valid_idxs], t[test_idxs]
    n_prov = len(np.unique(d))

    # Get outcome model
    if outcome_model_class is None:
        outcome_model_class = secondary_class

    outcome_model = get_outcome_model(outcome_model_class, X_train, X_valid, t_train, t_valid)
    if outcome_model_class == 'LogisticRegression':
        print('Outcome model coefficients and intercept')
        print(outcome_model['logReg'].coef_)
        print(outcome_model['logReg'].intercept_)
    elif outcome_model_class == 'RandomForest':
        pdp_features = range(X_train.shape[1])
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(nrows=2, ncols=int((len(pdp_features)+1)/2))
        PartialDependenceDisplay.from_estimator(outcome_model, X_train, pdp_features, feature_names=feature_names, ax=ax)
        plt.tight_layout()
        plt.savefig(pdp_filename)

    # Take cached results if they are available.
    if filename is not None and plot_filename is not None and os.path.isfile(filename) and os.path.isfile(plot_filename):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)
    else:
        if model_class == 'CausalForest':
            assert oracle_preds is not None

        # Hardcode random seed.
        np.random.seed(1891)

        if model_class == 'ULearner':
            results_dict = run_U_learner(secondary_class, X, d, t, train_idxs, valid_idxs, test_idxs, 
                                         filename, true_region_func, true_provider_split, beta)
        elif model_class == 'CausalForest':
            results_dict = run_causal_forest(X, d, t, train_idxs, valid_idxs, test_idxs, oracle_preds, 
                                             filename, true_region_func, true_provider_split, beta)
        elif model_class == 'Iterative':
            results_dict = run_iterative_alg_tune_beta(secondary_class,
                                                       X_train, X_valid, X_test,
                                                       d_train, d_valid, d_test,
                                                       t_train, t_valid, t_test,
                                                       n_prov,
                                                       outcome_model,
                                                       filename,
                                                       true_region_func=true_region_func, 
                                                       true_provider_split=true_provider_split,
                                                       betas=beta, n_iter=n_iter)
            print('Difference between outcome model prediction and true outcome in region vs all samples')
            X_train_outcome_pred = outcome_model.predict_proba(X_train)[:,1]
            print('Average predicted training outcome in region: {0:.4f}'.format(np.mean(
                  X_train_outcome_pred[results_dict['train_region_idxs']])))
            print('Average training outcome in region: {0:.4f}'.format(np.mean(t_train[results_dict['train_region_idxs']])))
            print('Average predicted training outcome: {0:.4f}'.format(np.mean(X_train_outcome_pred)))
            print('Average training outcome: {0:.4f}'.format(np.mean(t_train)))
        elif model_class == 'Direct':
            results_dict = run_model_direct(secondary_class, X, d, t, train_idxs, valid_idxs, test_idxs, filename, 
                                            true_region_func, true_provider_split, beta)
        else:
            raise Exception('method_name not recognized.')

    # Write results to stdout
    if verbose:
        if true_region_func is not None:
            print('Region precision: ' + str(results_dict['region_precision']))
            print('Region recall: ' + str(results_dict['region_recall']))
            print('Region AUC: ' + str(results_dict['region_auc']))
        if results_dict['partition_acc'] is not None:
            print('Partition accuracy: ' + str(results_dict['partition_acc']))
        if 'time_taken' in results_dict and results_dict['time_taken'] is not None:
            print('Time taken: ' + str(results_dict['time_taken']))

    return results_dict
    
def run_realworld_experiment(datasource, model_class, secondary_class, outcome_model_class=None):
    '''
    Runs experiments for diabetes and Parkinson's
    
    Parameters
    ----------
    datasource: str
        'diabetes' or 'ppmi'
        
    model_class: str
        Name of model class, e.g. 'Iterative', 'CausalForest', 'ULearner', 'Direct', 'TARNet'
        
    secondary_class: str
        Name of region and outcome model class, e.g. 'DecisionTree', 'LogisticRegression', 'RandomForest'
    
    outcome_model_class: str
        Name of outcome model class, e.g. 'DecisionTree', 'LogisticRegression', 'RandomForest'
        Overrides previous parameter if not None
    '''
    
    assert datasource in {'diabetes', 'ppmi'}
    # Load the correct dataset.
    if datasource == 'diabetes':
        X, d, t, train_fold_idxs, valid_fold_idxs, test_idxs, orig_prvs, scaler = load_diabetes_data()
        beta = [0.25]
        region_X_feat_idxs = [0,1,2]#,3] # removing treatment date from region model
        feature_names = ['egfr','creatinine','heart_disease','treatment_date_sec']
    else: # datasource == 'ppmi':
        X, d, t, train_fold_idxs, valid_fold_idxs, test_idxs, orig_prvs, scaler = load_ppmi_data()
        beta = [0.25]
        region_X_feat_idxs = [0,1,2]
        feature_names = ['age','disdur','mds23']
    
    if model_class == 'CausalForest':
        ulearner_fold_results_dict = dict()
        for fold_idx in range(4):
            ulearner_fold_results_dict[fold_idx] = dict()

    # Run algorithms on each fold of the data
    fold_results_dict = dict()
    for fold_idx in range(4):
        fold_results_dict[fold_idx] = dict()
        train_idxs = train_fold_idxs[fold_idx]
        valid_idxs = valid_fold_idxs[fold_idx]

        # Separate handling for causal forest baselines
        oracle_preds = None
        if model_class == 'CausalForest':
            ulearner_model_class = 'ULearner' + secondary_class
            ulearner_filename = filename_prefix + datasource + '_' + ulearner_model_class + '_fold' + str(fold_idx) + '.pkl'
            ulearner_results_dict = run_model('ULearner', secondary_class, X, d, t, train_idxs, valid_idxs, test_idxs, 
                                              ulearner_filename, beta=beta)
            ulearner_fold_results_dict[fold_idx]['train_idxs'] = train_idxs
            ulearner_fold_results_dict[fold_idx]['valid_idxs'] = valid_idxs
            ulearner_fold_results_dict[fold_idx]['test_idxs'] = test_idxs
            ulearner_fold_results_dict[fold_idx]['d'] = d
            ulearner_fold_results_dict[fold_idx]['train_region_idxs'] = ulearner_results_dict['train_region_idxs']
            ulearner_fold_results_dict[fold_idx]['valid_region_idxs'] = ulearner_results_dict['valid_region_idxs']
            ulearner_fold_results_dict[fold_idx]['test_region_idxs'] = ulearner_results_dict['test_region_idxs']
            ulearner_fold_results_dict[fold_idx]['pred_provider_split'] = ulearner_results_dict['pred_provider_split']
            ulearner_fold_results_dict[fold_idx]['pred_provider_split_test'] = ulearner_results_dict['pred_provider_split_test']
            if datasource == 'diabetes':
                process_diabetes_results(datasource, ulearner_model_class, secondary_class, ulearner_results_dict, X, d, t, 
                                         train_idxs, valid_idxs, test_idxs, orig_prvs, scaler, fold_idx)
            else: # datasource == 'ppmi':
                process_ppmi_results(ulearner_results_dict, X, d, t, train_idxs, valid_idxs, test_idxs, orig_prvs, 
                                     ulearner_model_class, scaler, fold_idx)
            oracle_preds = ulearner_results_dict['all_resids_pred']

        # Example filename: saved/diabetes_IterativeDecisionTree_fold0.pkl
        filename = filename_prefix + datasource + '_' + model_class + secondary_class + '_fold' + str(fold_idx) + '.pkl'

        results_dict = run_model(model_class, secondary_class, X, d, t, train_idxs, valid_idxs, test_idxs, filename, 
                                 oracle_preds=oracle_preds, beta=beta, outcome_model_class=outcome_model_class, 
                                 region_X_feat_idxs=region_X_feat_idxs, feature_names=feature_names, 
                                 pdp_filename=pdp_filename)

        # Further processing for iterative algorithm, e.g. visualize decision trees
        if model_class == 'Iterative':
            fold_results_dict[fold_idx]['train_idxs'] = train_idxs
            fold_results_dict[fold_idx]['valid_idxs'] = valid_idxs
            fold_results_dict[fold_idx]['test_idxs'] = test_idxs
            fold_results_dict[fold_idx]['d'] = d
            fold_results_dict[fold_idx]['train_region_idxs'] = results_dict['train_region_idxs']
            fold_results_dict[fold_idx]['valid_region_idxs'] = results_dict['valid_region_idxs']
            fold_results_dict[fold_idx]['test_region_idxs'] = results_dict['test_region_idxs']
            fold_results_dict[fold_idx]['pred_provider_split'] = results_dict['pred_provider_split']
            fold_results_dict[fold_idx]['pred_provider_split_test'] = results_dict['pred_provider_split_test']
            if datasource == 'diabetes':
                process_diabetes_results(datasource, model_class, secondary_class, results_dict, X, d, t, train_idxs, 
                                         valid_idxs, test_idxs, orig_prvs, scaler, fold_idx, outcome_model_class, 
                                         region_X_feat_idxs=region_X_feat_idxs)
            else:# datasource == 'ppmi':
                process_ppmi_results(results_dict, X, d, t, train_idxs, valid_idxs, test_idxs, orig_prvs, 
                                     model_class, scaler, fold_idx)

    # Assessing stability
    if model_class == 'Iterative':
        print(model_class)
        evaluate_fold_consistency(fold_results_dict, X, d, t)
        evaluate_heldout_fold_consistency(fold_results_dict, X, d, t)
        evaluate_test_region_fold_consistency(fold_results_dict, X, d, t)

if __name__ == "__main__":
    np.random.seed(1681)
    filename_prefix = 'saved/'

    # Expected format: <this script>.py <datasource> <model_class> <secondary_class> [<outcome_model_class>]
    # Last parameter optional. Only if different from secondary_class (region model)
    assert len(sys.argv) == 4

    datasource = sys.argv[1]

    assert datasource in ['diabetes', 'ppmi']

    model_class = sys.argv[2]
    assert model_class in ['Direct', 'CausalForest', 'ULearner', 'Iterative']

    secondary_class = sys.argv[3]
    if model_class != 'Direct':
        assert secondary_class in ['LogisticRegression', 'DecisionTree', 'RandomForest', 'Oracle']
    else:
        assert secondary_class in ['LogisticRegression', 'DecisionTree', 'RandomForest']
        
    if len(sys.argv) > 4:
        outcome_model_class = sys.argv[4]
        assert outcome_model_class in ['LogisticRegression', 'DecisionTree', 'RandomForest']

    run_realworld_experiment(datasource, model_class, secondary_class, outcome_model_class)
