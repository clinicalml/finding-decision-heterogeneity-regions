import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from baselines import best_G
from sklearn.tree import plot_tree, DecisionTreeRegressor

filename_prefix = 'saved/'

def process_diabetes_results(datasource, model_class, secondary_class, results_dict, X, d, t, train_idxs, valid_idxs, 
                             test_idxs, orig_sites, scaler, fold_idx, outcome_model_class=None, region_X_feat_idxs=None):
    '''
    Function to process diabetes results and create visualizations.
    Runs significance test
    Prints statistics for each node of region decision tree

    Parameters
    ==========
    datasource : str
        Which dataset is being used, e.g. 'diabetes'

    model_class : str
        Which model is being used, e.g. 'Iterative'

    secondary_class : str
        Which region_model is being used, e.g. 'DecisionTree'
        
    results_dict: dictionary
        Contains region model, region indices, outcome predictions, agent groupings, etc.
    
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
        
    orig_sites: list of strings
        Names of agents
    
    scaler: sklearn StandardScaler object
        Used to reverse normalization of X
        
    fold_idx: int
        Fold number
        
    outcome_model_class: str
        Name of outcome model class
        
    region_X_feat_idxs: list of ints
        Indices of features used in region model
    '''
    
    if outcome_model_class is None:
        outcome_model_class = secondary_class
    if region_X_feat_idxs is None:
        region_X_feat_idxs = range(X.shape[1])
    else:
        assert np.min(region_X_feat_idxs) >= 0
        assert np.max(region_X_feat_idxs) < X.shape[1]
        assert len(np.unique(region_X_feat_idxs)) == len(region_X_feat_idxs)

    status_line = "Processsing %s results for %s %s, fold %d" % (datasource, model_class, secondary_class, fold_idx)
    print("=" * len(status_line))
    print(status_line)
    print("=" * len(status_line))

    # Report the groupings of providers
    pred_provider_split = None
    if 'pred_provider_split' in results_dict.keys():
        pred_provider_split = results_dict['pred_provider_split']
        if pred_provider_split is None:
            return
    pred_provider_split_test = None
    if 'pred_provider_split_test' in results_dict.keys():
        pred_provider_split_test = results_dict['pred_provider_split_test']
        if pred_provider_split_test is None:
            return
    for i in range(2):
        group_idxs = np.nonzero(np.where(pred_provider_split_test == i, 1, 0))[0]
        group_sites = (np.array(orig_sites)[group_idxs]).tolist()
        # print('Group ' + str(i) + ' providers: ' + ', '.join([str(j) for j in group_sites]))

    # Identify data points in the test set
    test_region_idxs = test_idxs[results_dict['test_region_idxs']]
    Xt_test_pred = results_dict['Xt_test_pred']
    n_prov = results_dict['n_prov']

    X_test = X[test_idxs]
    t_test = t[test_idxs]
    d_test = d[test_idxs]
    X_test_unnorm = scaler.inverse_transform(X_test)
    X_test_unnorm[:,-1] = pd.to_datetime(X_test_unnorm[:,-1] * 1e9)
    
    # Identify data points in the test set
    test_region_idxs = test_idxs[results_dict['test_region_idxs']]
    Xt_test_pred = results_dict['Xt_test_pred']
    n_prov = results_dict['n_prov']

    X_test = X[test_idxs]
    t_test = t[test_idxs]
    d_test = d[test_idxs]
    X_test_unnorm = scaler.inverse_transform(X_test)
    X_test_unnorm = X_test_unnorm[:,region_X_feat_idxs]
    X_test = X_test[:,region_X_feat_idxs]
    if region_X_feat_idxs[-1] == 4:
        X_test_unnorm[:,-1] = pd.to_datetime(X_test_unnorm[:,-1] * 1e9) # treatment date
    
    # Identify data points in the test set, and in the region
    X_test_region = X[test_region_idxs]
    d_test_region = d[test_region_idxs]
    t_test_region = t[test_region_idxs]
    
    d_test_region_unique = np.unique(d_test_region).tolist()
    # print('Providers in test region: ' + ', '.join([str(i) for i in d_test_region_unique]))
    xlabels = ['eGFR', 'Creatinine', 'Heart disease', 'Treatment date']
    feature_for_filename = ['egfr', 'creatinine', 'heart_disease', 'treatment_date_sec']
    X_test_region_unnorm = scaler.inverse_transform(X_test_region)
    X_test_region_unnorm = X_test_region_unnorm[:,region_X_feat_idxs]
    X_test_region = X_test_region[:,region_X_feat_idxs]
    if region_X_feat_idxs[-1] == 3: # treatment date
        X_test_region_unnorm[:,-1] = pd.to_datetime(X_test_region_unnorm[:,-1] * 1e9)
    
    group_test = np.zeros(len(test_idxs))
    for i in range(len(test_idxs)):
        group_test[i] = pred_provider_split_test[int(d_test[i])]

    # For each provider, count the number of samples they have in the region, in the test set
    group_test_region = np.empty(d_test_region.shape)
    pred_provider_split_test_freq = np.zeros(len(pred_provider_split_test))
    for i in range(len(d_test_region)):
        group_test_region[i] = pred_provider_split_test[int(d_test_region[i])]
        pred_provider_split_test_freq[int(d_test_region[i])] += 1
        
    # Only take providers with at least 2 samples in the region, in the test set
    pred_provider_split_test_mask = pred_provider_split_test_freq >= 2
    mask_test_region = pred_provider_split_test_mask[d_test_region]
    group0_idxs = np.nonzero(np.where((group_test_region == 0) & (mask_test_region), 1, 0))[0]
    group0_X_test_region_unnorm = X_test_region_unnorm[group0_idxs]
    group0_t_test_region = t_test_region[group0_idxs]
    group1_idxs = np.nonzero(np.where((group_test_region == 1) & (mask_test_region), 1, 0))[0]
    group1_X_test_region_unnorm = X_test_region_unnorm[group1_idxs]
    group1_t_test_region = t_test_region[group1_idxs]

    # Plotting parameters
    treatment_classes = ['metformin', 'sitagliptin or sulfonylurea']
    colors = ['blue', 'orange']

    # Run a test to check if the identified region is better than a randomly selected region
    if model_class.startswith("Iterative"):

        print()
        print("Test against random regions")
        train_scores = results_dict['train_scores']
        valid_scores = results_dict['valid_scores']
        train_region_score = np.mean(np.concatenate([train_scores[results_dict['train_region_idxs']], 
                                                     valid_scores[results_dict['valid_region_idxs']]]))
        print("Q(S, G) on train+valid set: %.4f" % train_region_score)

        test_scores = results_dict['test_scores']
        test_region_score = np.mean(test_scores[results_dict['test_region_idxs']])
        print("Q(S, G_{test, max}) on test set: %.4f" % test_region_score)

        rand_test_iter = 100
        rand_test_scores = np.zeros(rand_test_iter)
        beta = len(test_region_idxs)/len(d_test)
        for i in range(rand_test_iter):
            rand_region_boolean = np.random.choice([0, 1], size=len(d_test), p=[1-beta, beta]).astype(bool)
            _, rand_test_score = best_G(rand_region_boolean, X_test, t_test, d_test, Xt_test_pred, n_prov)
            rand_test_scores[i] = rand_test_score
        print("Q(S_rand, G_{rand, max}) on test set: %.4f (%.4f)" % (np.mean(rand_test_scores), np.std(rand_test_scores)))

        group_train_region = np.array([pred_provider_split[d_test_region[i]] for i in range(len(d_test_region))])
        test_region_train_score = np.mean((t_test_region - Xt_test_pred[results_dict['test_region_idxs']]) * (group_train_region))
        print("Q(S, G_{train, max}) on test set: %.4f" % (test_region_train_score))

        rand_test_scores = np.zeros(rand_test_iter)
        beta = np.sum(pred_provider_split_test) / len(pred_provider_split_test)
        for i in range(rand_test_iter):
            rand_grouping_boolean = np.random.choice([0, 1], size=n_prov, p=[1-beta, beta]).astype(bool)
            group_rand_region = np.array([rand_grouping_boolean[d_test_region[i]] for i in range(len(d_test_region))])
            assert len(group_rand_region) == len(t_test_region)
            assert len(results_dict['test_region_idxs']) == len(t_test_region)
            rand_test_score = np.mean((t_test_region - Xt_test_pred[results_dict['test_region_idxs']]) * (group_rand_region))
            rand_test_scores[i] = rand_test_score
        print("Q(S, G_rand) on test set: %.4f (%.4f)" % (np.mean(rand_test_scores), np.std(rand_test_scores)))


    # If the region model is a decision tree, generate visualizations
    if 'region_model' in results_dict.keys() and isinstance(results_dict['region_model'], DecisionTreeRegressor):
        print()
        print("====================")
        print("Decision tree output")
        print("====================")

        dt = results_dict['region_model']
        cutoff = results_dict['cutoff']
        print("Cutoff: %.4f" % cutoff)
        
        plt.figure(figsize=(20, 10))
        plot_tree(dt, feature_names=xlabels, fontsize=10, filled=True, node_ids=True)
        fname = filename_prefix + datasource + '_' + model_class + secondary_class + '_TRAIN_fold' + str(fold_idx) + '.pdf'
        plt.savefig(fname, dpi=1000, bbox_inches='tight') 
        plt.close()
        print("Created visualization of decision tree at %s" % fname)
        
        # Honesty: compute numbers on held-out data. Modifies the decision tree
        test_region_leaf_idxs = dt.apply(X[test_region_idxs])
        test_region_scores = results_dict['test_scores'][results_dict['test_region_idxs']]

        test_leaves = dt.apply(X[test_idxs])
        test_scores = results_dict['test_scores']
        test_region_leaf_values = []
        n_nodes = len(dt.tree_.feature)

        plt.rcParams.update({'font.size': 15})

        for i in range(n_nodes):
            if dt.tree_.children_left[i] != -1:
                continue

            test_leaf_idxs = np.nonzero(test_leaves == i)[0]
            if len(test_leaf_idxs) > 0:

                print()
                print("Node %d" % i)

                # Determine if this leaf is in the region
                if dt.tree_.value[i, 0, 0] >= cutoff:
                    print("In region? Yes")
                else:
                    print("In region? No")

                # Compute the quantities Q(S, G) on the test set
                honest_leaf_score = np.mean(test_scores[test_leaves == i])
                mask_test_leaf = pred_provider_split_test_mask[d_test[test_leaf_idxs]]

                group0_leaf_idxs = np.nonzero(np.where((group_test[test_leaf_idxs] == 0) & \
                                                       (mask_test_leaf), 1, 0))[0]
                group1_leaf_idxs = np.nonzero(np.where((group_test[test_leaf_idxs] == 1) & \
                                                       (mask_test_leaf), 1, 0))[0]
                if len(group0_leaf_idxs) != 0:
                    print("Group 0: positive: %d, total: %d, fraction: %.2f" % \
                          (np.sum(t_test[test_leaf_idxs][group0_leaf_idxs]), len(group0_leaf_idxs), 
                           np.sum(t_test[test_leaf_idxs][group0_leaf_idxs])/len(group0_leaf_idxs)))
                if len(group1_leaf_idxs) != 0:
                    print("Group 1: positive: %d, total: %d, fraction: %.2f" % \
                          (np.sum(t_test[test_leaf_idxs][group1_leaf_idxs]), len(group1_leaf_idxs), 
                           np.sum(t_test[test_leaf_idxs][group1_leaf_idxs])/len(group1_leaf_idxs)))

                X_test_leaf_unnorm = X_test_unnorm[test_leaf_idxs]
                group0_X_test_leaf_unnorm = X_test_unnorm[test_leaf_idxs][group0_leaf_idxs]
                group0_t_test_leaf = t_test[test_leaf_idxs][group0_leaf_idxs]
                group1_X_test_leaf_unnorm = X_test_unnorm[test_leaf_idxs][group1_leaf_idxs]
                group1_t_test_leaf = t_test[test_leaf_idxs][group1_leaf_idxs]

                # Visualize variation in each node. See Figure 7-5 in thesis, or Figure 5 in KDD submission.
                for j in range(X_test_leaf_unnorm.shape[1]):
                    plt.clf()
                    Xjmin = np.min(X_test_leaf_unnorm[mask_test_leaf,j])
                    Xjmax = np.max(X_test_leaf_unnorm[mask_test_leaf,j])

                    # Post-hoc plotting parameters
                    if feature_for_filename[j] == 'creatinine':
                        Xjmax = min(2.0, Xjmax)
                    if feature_for_filename[j] == 'egfr':
                        Xjmin = max(30, Xjmin)

                    fig, host = plt.subplots(figsize=(8,5))
                    par1 = host.twinx()

                    group0_t0_idxs = np.nonzero(np.where(group0_t_test_leaf == 0, 1, 0))[0]
                    group0_t1_idxs = np.nonzero(np.where(group0_t_test_leaf == 1, 1, 0))[0]
                    group0_t0_X_test_leaf_unnorm = group0_X_test_leaf_unnorm[group0_t0_idxs,j]
                    group0_t1_X_test_leaf_unnorm = group0_X_test_leaf_unnorm[group0_t1_idxs,j]

                    n_bins = 10
                    hist0, bins = np.histogram(group0_t0_X_test_leaf_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
                    hist1, _ = np.histogram(group0_t1_X_test_leaf_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
                    group0_hist = 100 * (1-np.array([hist1[i]/(hist1[i]+hist0[i]) if hist1[i]+hist0[i]>0 else 0 for i in range(len(hist0))]))
                    
                    group1_t0_idxs = np.nonzero(np.where(group1_t_test_leaf == 0, 1, 0))[0]
                    group1_t1_idxs = np.nonzero(np.where(group1_t_test_leaf == 1, 1, 0))[0]
                    group1_t0_X_test_leaf_unnorm = group1_X_test_leaf_unnorm[group1_t0_idxs,j]
                    group1_t1_X_test_leaf_unnorm = group1_X_test_leaf_unnorm[group1_t1_idxs,j]

                    hist0, _ = np.histogram(group1_t0_X_test_leaf_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
                    hist1, _ = np.histogram(group1_t1_X_test_leaf_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
                    group1_hist = 100 * (1-np.array([hist1[i]/(hist1[i]+hist0[i]) if hist1[i]+hist0[i]>0 else 0 for i in range(len(hist0))]))
                    
                    if feature_for_filename[j] == 'treatment_date_sec':
                        group0_X_test_leaf_unnorm_date = pd.to_datetime(group0_X_test_leaf_unnorm[:, j])
                        group0_X_test_leaf_unnorm_date = pd.to_datetime(group0_X_test_leaf_unnorm[:, j])
                        Xjmin_date = pd.to_datetime(Xjmin)
                        Xjmax_date = pd.to_datetime(Xjmax)
                        
                        group1_X_test_leaf_unnorm_date = pd.to_datetime(group1_X_test_leaf_unnorm[:, j])
                        group1_X_test_leaf_unnorm_date = pd.to_datetime(group1_X_test_leaf_unnorm[:, j])
                        _, bins, _ = par1.hist([group0_X_test_leaf_unnorm_date, group1_X_test_leaf_unnorm_date], 
                                               bins=n_bins, range=[Xjmin_date, Xjmax_date], 
                                               alpha=.3, color=colors, label=['G=0, freq', 'G=1, freq'])
                        host.plot([0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)], group0_hist, color=colors[0], lw=2, 
                                  label='G=0, % MET')
                        host.plot([0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)], group1_hist, color=colors[1], lw=2, 
                                  label='G=1, % MET')
                    else:
                        _, bins, _ = par1.hist([group0_X_test_leaf_unnorm[:, j], group1_X_test_leaf_unnorm[:, j]], 
                                               bins=n_bins, range=[Xjmin, Xjmax], 
                                               alpha=.3, color=colors, label=['G=0, freq', 'G=1, freq'])
                        host.plot([0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)], group0_hist, color=colors[0], lw=2, 
                                  label='G=0, % MET')
                        host.plot([0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)], group1_hist, color=colors[1], lw=2, 
                                  label='G=1, % MET')
                        
                    if i == 1 and feature_for_filename[j] == 'treatment_date_sec':
                        lines, labels = host.get_legend_handles_labels()
                        lines2, labels2 = par1.get_legend_handles_labels()
                        plt.legend(lines + lines2, labels + labels2, ncol=1)

                    if feature_for_filename[j] == 'treatment_date_sec':
                        years = mdates.YearLocator()
                        years_fmt = mdates.DateFormatter('%Y')
                        host.xaxis.set_major_locator(years)
                        host.xaxis.set_major_formatter(years_fmt)
                    host.set_xlabel(xlabels[j])
                    host.set_ylabel("Percentage Metformin")
                    par1.set_ylabel("Frequency")

                    plt.tight_layout()
                    plt.title("Node %d" % i)
                    fname = filename_prefix + datasource + '_' + model_class + '_DecisionTree_node' + str(i) + '_' + feature_for_filename[j] + '_fold' + str(fold_idx) + '.pdf'
                    plt.savefig(fname, bbox_inches='tight')     
                    plt.close()
                    print("Created visualization for this node at %s" % fname)

            else:
                honest_leaf_score = 0

            # Set the Q(S, G) values in dt.tree_ to their values on the test set.
            dt.tree_.value[i, 0, 0] = honest_leaf_score
            test_region_leaf_values.append(honest_leaf_score)
            
        # Rescale the thresholds back to their original scales
        for i in range(n_nodes):
            if dt.tree_.children_left[i] == -1:
                continue
            ix = dt.tree_.feature[i]
            dt.tree_.threshold[i] = dt.tree_.threshold[i]*scaler.scale_[ix] + scaler.mean_[ix]
            '''
            if feature_for_filename[ix] == 'treatment_date_sec':
                date = pd.to_datetime(dt.tree_.threshold[i] * 1e9)
                print("Conversion: %s to %s" % (dt.tree_.threshold[i], date))
            '''
        
        # Plot the decision tree with the rescaled thresholds and 
        plt.figure(figsize=(20, 10))
        plot_tree(dt, feature_names=xlabels, fontsize=10, filled=True, node_ids=True)
        fname = filename_prefix + datasource + '_' + model_class + '_DecisionTree_fold' + str(fold_idx) + '.pdf'
        plt.savefig(filename_prefix + datasource + '_' + model_class + '_DecisionTree_fold' + str(fold_idx) + '.pdf', 
                    dpi=1000, bbox_inches='tight') 
        plt.close()
        print()
        print("Created visualization of decision tree with Q(S, G) values at %s" % fname)

def process_ppmi_results(results_dict, X, d, t, train_idxs, valid_idxs, test_idxs, orig_sites, model_class, scaler, fold_idx):
    '''
    Function to process Parkinson's results and create visualizations.
    Runs significance test
    Prints statistics for each node of region decision tree

    Parameters
    ==========
    results_dict: dictionary
        Contains region model, region indices, outcome predictions, agent groupings, etc.
    
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
        
    orig_sites: list of strings
        Names of agents
        
    model_class: str
        Name of region and outcome model class
    
    scaler: sklearn StandardScaler object
        Used to reverse normalization of X
        
    fold_idx: int
        Fold number
    '''
    pred_provider_split_test = None
    if 'pred_provider_split_test' in results_dict.keys():
        pred_provider_split_test = results_dict['pred_provider_split_test']
    if pred_provider_split_test is None:
        return
    for i in range(2):
        group_idxs = np.nonzero(np.where(pred_provider_split_test == i, 1, 0))[0]
        group_sites = (np.array(orig_sites)[group_idxs]).tolist()
        print('Group ' + str(i) + ' providers: ' + ', '.join([str(j) for j in group_sites]))
    test_region_idxs = test_idxs[results_dict['test_region_idxs']]
    X_test_region = X[test_region_idxs]
    d_test_region = d[test_region_idxs]
    t_test_region = t[test_region_idxs]
    d_test_region_unique = np.unique(d_test_region).tolist()
    print('Providers in region: ' + ', '.join([str(orig_sites[int(i)]) for i in d_test_region_unique]))
    X_test = X[test_idxs]
    t_test = t[test_idxs]
    d_test = d[test_idxs]

    
    group_test = np.zeros(len(test_idxs))
    for i in range(len(test_idxs)):
        group_test[i] = pred_provider_split_test[int(d_test[i])]

    
    group_test_region = np.empty(d_test_region.shape)
    xlabels = ['Age', 'Disease duration (years)', 'MDS-UPDRS II + III']
    feature_for_filename = ['age', 'disdur', 'mds23']
    X_test_region_unnorm = scaler.inverse_transform(X_test_region)
    pred_provider_split_test_freq = np.zeros(len(pred_provider_split_test))
    for i in range(len(d_test_region)):
        group_test_region[i] = pred_provider_split_test[int(d_test_region[i])]
        pred_provider_split_test_freq[int(d_test_region[i])] += 1
        
    # Only take providers with at least 2 samples
    pred_provider_split_test_mask = pred_provider_split_test_freq >= 2
    mask_test_region = pred_provider_split_test_mask[d_test_region]
    group0_idxs = np.nonzero(np.where((group_test_region == 0) & (mask_test_region), 1, 0))[0]
    group0_X_test_region_unnorm = X_test_region_unnorm[group0_idxs]
    group0_t_test_region = t_test_region[group0_idxs]
    group1_idxs = np.nonzero(np.where((group_test_region == 1) & (mask_test_region), 1, 0))[0]
    group1_X_test_region_unnorm = X_test_region_unnorm[group1_idxs]
    group1_t_test_region = t_test_region[group1_idxs]
    treatment_classes = ['rasagiline', 'levodopa']
    colors = ['blue', 'orange']
    plt.rcParams.update({'font.size': 15})
    for j in range(X_test_region.shape[1]):
        plt.clf()
        if set([int(i) for i in np.unique(X_test_region_unnorm[:,j]).tolist()]).issubset({0,1}):
            group0_t0_feat0_count = np.sum(np.where(np.logical_and(group0_t_test_region==0, 
                                                                   group0_X_test_region_unnorm[:,j]==0), 1, 0))
            group0_t1_feat0_count = np.sum(np.where(np.logical_and(group0_t_test_region==1, 
                                                                   group0_X_test_region_unnorm[:,j]==0), 1, 0))
            group0_t0_feat1_count = np.sum(np.where(np.logical_and(group0_t_test_region==0, 
                                                                   group0_X_test_region_unnorm[:,j]==1), 1, 0))
            group0_t1_feat1_count = np.sum(np.where(np.logical_and(group0_t_test_region==1, 
                                                                   group0_X_test_region_unnorm[:,j]==1), 1, 0))
            group1_t0_feat0_count = np.sum(np.where(np.logical_and(group1_t_test_region==0, 
                                                                   group1_X_test_region_unnorm[:,j]==0), 1, 0))
            group1_t1_feat0_count = np.sum(np.where(np.logical_and(group1_t_test_region==1, 
                                                                   group1_X_test_region_unnorm[:,j]==0), 1, 0))
            group1_t0_feat1_count = np.sum(np.where(np.logical_and(group1_t_test_region==0, 
                                                                   group1_X_test_region_unnorm[:,j]==1), 1, 0))
            group1_t1_feat1_count = np.sum(np.where(np.logical_and(group1_t_test_region==1, 
                                                                   group1_X_test_region_unnorm[:,j]==1), 1, 0))
            plt.bar([-0.15,0.05,0.85,1.05], 
                    [group0_t0_feat0_count, group0_t1_feat0_count, group0_t0_feat1_count, group0_t1_feat1_count], 
                    width=0.1, color=colors[0], label='Group 0')
            plt.bar([-0.05,0.15,0.95,1.15], 
                    [group1_t0_feat0_count, group1_t1_feat0_count, group1_t0_feat1_count, group1_t1_feat1_count], 
                    width=0.1, color=colors[1], label='Group 1')
            plt.legend()
            plt.xticks([0,1])
            plt.xlabel(feature_for_filename[j])
            fig = plt.gcf()
            fig.set_figheight(2)
        else:
            Xjmin = np.min(X_test_region_unnorm[mask_test_region,j])
            Xjmax = np.max(X_test_region_unnorm[mask_test_region,j])
                
            fig, host = plt.subplots(figsize=(8,5))
            par1 = host.twinx()
            
            # Group 0
            group0_t0_idxs = np.nonzero(np.where(group0_t_test_region == 0, 1, 0))[0]
            group0_t1_idxs = np.nonzero(np.where(group0_t_test_region == 1, 1, 0))[0]
            group0_t0_X_test_region_unnorm = group0_X_test_region_unnorm[group0_t0_idxs,j]
            group0_t1_X_test_region_unnorm = group0_X_test_region_unnorm[group0_t1_idxs,j]
            
            n_bins = 5
            hist0, bins = np.histogram(group0_t0_X_test_region_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
            hist1, _ = np.histogram(group0_t1_X_test_region_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
            group0_hist = 100*np.array([hist1[i]/(hist1[i]+hist0[i]) if hist1[i]+hist0[i]>0 else 0 for i in range(len(hist0))])
            
            host.plot([0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)], group0_hist, color=colors[0], lw=2, 
                      label='G=0, % T=L')
            group1_t0_idxs = np.nonzero(np.where(group1_t_test_region == 0, 1, 0))[0]
            group1_t1_idxs = np.nonzero(np.where(group1_t_test_region == 1, 1, 0))[0]
            group1_t0_X_test_region_unnorm = group1_X_test_region_unnorm[group1_t0_idxs,j]
            group1_t1_X_test_region_unnorm = group1_X_test_region_unnorm[group1_t1_idxs,j]
            
            hist0, _ = np.histogram(group1_t0_X_test_region_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
            hist1, _ = np.histogram(group1_t1_X_test_region_unnorm, bins=n_bins, range=[Xjmin, Xjmax])
            group1_hist = 100*np.array([hist1[i]/(hist1[i]+hist0[i]) if hist1[i]+hist0[i]>0 else 0 for i in range(len(hist0))])
            
            _, bins, _ = par1.hist([group0_X_test_region_unnorm[:, j], group1_X_test_region_unnorm[:, j]], 
                                               bins=n_bins, range=[Xjmin, Xjmax], 
                                               alpha=.3, color=colors, label=['G=0, freq', 'G=1, freq'])
            host.plot([0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)], group0_hist, color=colors[0], lw=2, 
                      label='G=0, % T=L')
            host.plot([0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)], group1_hist, color=colors[1], lw=2, 
                      label='G=1, % T=L')
            
            if j == X_test_region.shape[1] - 1:
                lines, labels = host.get_legend_handles_labels()
                lines2, labels2 = par1.get_legend_handles_labels()
                plt.legend(lines + lines2, labels + labels2, ncol=1)
            
            host.set_xlabel(xlabels[j])
            host.set_ylabel("Percentage Levodopa")
            par1.set_ylabel("Frequency")
            
        plt.tight_layout()
        plt.savefig(filename_prefix + 'ppmi_' + model_class + '_'  + feature_for_filename[j] + '_fold' + str(fold_idx) + '.pdf')     
           
    
    if model_class.startswith("Iterative"):
        train_scores = results_dict['train_scores']
        valid_scores = results_dict['valid_scores']
        train_region_score = np.mean(np.concatenate([train_scores[results_dict['train_region_idxs']], 
                                                     valid_scores[results_dict['valid_region_idxs']]]))
        print("Q(S, G) on train+valid set: %.4f" % train_region_score)
        
        test_scores = results_dict['test_scores']
        test_region_score = np.mean(test_scores[results_dict['test_region_idxs']])
        print("Q(S, G_test) on test set: %.4f" % test_region_score)
        
        rand_test_iter = 100
        rand_test_scores = np.zeros(rand_test_iter)
        X_test = X[test_idxs]
        t_test = t[test_idxs]
        d_test = d[test_idxs]
        Xt_test_pred = results_dict['Xt_test_pred']
        n_prov = results_dict['n_prov']
        beta = len(test_region_idxs)/len(d_test) 
        for i in range(rand_test_iter):
            rand_region_boolean = np.random.choice([0, 1], size=len(d_test), p=[1-beta, beta]).astype(bool)
            _, rand_test_score = best_G(rand_region_boolean, X_test, t_test, d_test, Xt_test_pred, n_prov)
            rand_test_scores[i] = rand_test_score
        print("Q(S_rand, G_rand) on test set: %.4f (%.4f)" % (np.mean(rand_test_scores), np.std(rand_test_scores)))
    
        
    if 'region_model' in results_dict.keys() and isinstance(results_dict['region_model'], DecisionTreeRegressor):
        dt = results_dict['region_model']
        
        test_leaves = dt.apply(X[test_idxs])

        # Honesty: compute numbers on held-out data. Modifies the decision tree
        test_region_leaf_idxs = dt.apply(X[test_region_idxs])
        test_region_scores = results_dict['test_scores'][results_dict['test_region_idxs']]
        test_region_leaf_values = []
        n_nodes = len(dt.tree_.feature)
        
        for i in range(n_nodes):
            if dt.tree_.children_left[i] != -1:
                continue
            
            test_leaf_idxs = np.nonzero(test_leaves == i)[0]
            if len(test_leaf_idxs) > 0:
                
                honest_leaf_score = np.mean(test_scores[test_leaves == i])
                
                mask_test_leaf = pred_provider_split_test_mask[d_test[test_leaf_idxs]]
                
                group0_leaf_idxs = np.nonzero(np.where((group_test[test_leaf_idxs] == 0) & \
                                                       (mask_test_leaf), 1, 0))[0]
                group1_leaf_idxs = np.nonzero(np.where((group_test[test_leaf_idxs] == 1) & \
                                                       (mask_test_leaf), 1, 0))[0]
                if len(group0_leaf_idxs) != 0:
                    print("Node %d, group 0 positive: %d, group 0 total: %d, group 0 fraction: %.2f" % \
                          (i, np.sum(t_test[test_leaf_idxs][group0_leaf_idxs]), len(group0_leaf_idxs), \
                           np.sum(t_test[test_leaf_idxs][group0_leaf_idxs])/len(group0_leaf_idxs)))
                if len(group1_leaf_idxs) != 0:
                    print("Node %d, group 1 positive: %d, group 1 total: %d, group 1 fraction: %.2f" % \
                      (i, np.sum(t_test[test_leaf_idxs][group1_leaf_idxs]), len(group1_leaf_idxs), \
                       np.sum(t_test[test_leaf_idxs][group1_leaf_idxs])/len(group1_leaf_idxs)))

        for i in range(n_nodes):
            if dt.tree_.children_left[i] != -1:
                continue
                
            if np.sum(test_region_leaf_idxs == i) > 0:
                honest_leaf_score = np.mean(test_region_scores[test_region_leaf_idxs == i])
            else:
                honest_leaf_score = 0
            dt.tree_.value[i, 0, 0] = honest_leaf_score
            test_region_leaf_values.append(honest_leaf_score)
        
        for i in range(n_nodes):
            if dt.tree_.children_left[i] != -1:
                continue
            group0_leaf_idxs = np.nonzero(np.where((group_test_region == 0) & (mask_test_region) & (test_region_leaf_idxs == i), 1, 0))[0]
            group1_leaf_idxs = np.nonzero(np.where((group_test_region == 1) & (mask_test_region) & (test_region_leaf_idxs == i), 1, 0))[0]
            if len(group0_leaf_idxs) != 0:
                print("Node %d, group 0 positive: %d, group 0 total: %d, group 0 fraction: %.2f" % \
                      (i, np.sum(t_test_region[group0_leaf_idxs]), len(group0_leaf_idxs), \
                       np.sum(t_test_region[group0_leaf_idxs])/len(group0_leaf_idxs)))
            if len(group1_leaf_idxs) != 0:
                print("Node %d, group 1 positive: %d, group 1 total: %d, group 1 fraction: %.2f" % \
                  (i, np.sum(t_test_region[group1_leaf_idxs]), len(group1_leaf_idxs), \
                   np.sum(t_test_region[group1_leaf_idxs])/len(group1_leaf_idxs)))

        for i in range(n_nodes):
            if dt.tree_.children_left[i] == -1:
                continue
            ix = dt.tree_.feature[i]
            dt.tree_.threshold[i] = dt.tree_.threshold[i]*scaler.scale_[ix] + scaler.mean_[ix]
            if feature_for_filename[ix] == 'treatment_date_sec':
                date = pd.to_datetime(dt.tree_.threshold[i] * 1e9)
                print("Conversion: %s to %s" % (dt.tree_.threshold[i], date))
        
        plt.figure(figsize=(20, 10))
        plot_tree(dt, feature_names=xlabels, fontsize=10, filled=True, node_ids=True)
        plt.savefig(filename_prefix + 'ppmi_' + model_class + '_DecisionTree_fold' + str(fold_idx) + '.pdf', dpi=1000, bbox_inches='tight')     
        
def evaluate_heldout_fold_consistency(fold_results_dict, X, d, t):
    '''
    Evaluate consistency of whether a point is in the region when it is part of a validation vs training set

    Parameters
    ==========
    folds_results_dict: dictionary
        Contains keys 0, 1, 2, 3 mapped to the results dictionary for that fold.
        Results dictionary contains region model, region indices, outcome predictions, agent groupings, etc.
        
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
    ''' 
    train_valid_idxs = np.concatenate((fold_results_dict[0]['train_idxs'], fold_results_dict[0]['valid_idxs']))
    num_train_in_region = np.zeros(train_valid_idxs.shape)
    num_valid = np.zeros(train_valid_idxs.shape)
    num_valid_in_region = np.zeros(train_valid_idxs.shape)
    for i in range(len(train_valid_idxs)):
        idx = train_valid_idxs[i]
        for fold_idx in range(4):
            valid_idxs = fold_results_dict[fold_idx]['valid_idxs']
            if np.sum(np.where(valid_idxs==idx, 1, 0)) == 1:
                num_valid[i] += 1
                idx_in_valid = np.nonzero(np.where(valid_idxs==idx, 1, 0))[0][0]
                if np.sum(np.where(fold_results_dict[fold_idx]['valid_region_idxs'] == idx_in_valid, 1, 0)) == 1:
                    num_valid_in_region[i] += 1
            else:
                idx_in_train = np.nonzero(np.where(fold_results_dict[fold_idx]['train_idxs']==idx, 1, 0))[0][0]
                if np.sum(np.where(fold_results_dict[fold_idx]['train_region_idxs'] == idx_in_train, 1, 0)) == 1:
                    num_train_in_region[i] += 1
    vs1_vr0_tr0 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==0, num_train_in_region==0)), 1, 0))
    print('# points in 1 valid set, in 0 valid regions, in 0 train regions: ' + str(vs1_vr0_tr0))
    vs1_vr0_tr1 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==0, num_train_in_region==1)), 1, 0))
    print('# points in 1 valid set, in 0 valid regions, in 1 train region: ' + str(vs1_vr0_tr1))
    vs1_vr0_tr2 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==0, num_train_in_region==2)), 1, 0))
    print('# points in 1 valid set, in 0 valid regions, in 2 train regions: ' + str(vs1_vr0_tr2))
    vs1_vr0_tr3 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==0, num_train_in_region==3)), 1, 0))
    print('# points in 1 valid set, in 0 valid regions, in 3 train regions: ' + str(vs1_vr0_tr3))
    vs1_vr1_tr0 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==1, num_train_in_region==0)), 1, 0))
    print('# points in 1 valid set, in 1 valid region, in 0 train regions: ' + str(vs1_vr1_tr0))
    vs1_vr1_tr1 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==1, num_train_in_region==1)), 1, 0))
    print('# points in 1 valid set, in 1 valid region, in 1 train region: ' + str(vs1_vr1_tr1))
    vs1_vr1_tr2 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==1, num_train_in_region==2)), 1, 0))
    print('# points in 1 valid set, in 1 valid region, in 2 train regions: ' + str(vs1_vr1_tr2))
    vs1_vr1_tr3 = np.sum(np.where(np.logical_and.reduce((num_valid==1, num_valid_in_region==1, num_train_in_region==3)), 1, 0))
    print('# points in 1 valid set, in 1 valid region, in 3 train regions: ' + str(vs1_vr1_tr3))
    vs2_vr0_tr0 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==0, num_train_in_region==0)), 1, 0))
    print('# points in 2 valid sets, in 0 valid regions, in 0 train regions: ' + str(vs2_vr0_tr0))
    vs2_vr0_tr1 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==0, num_train_in_region==1)), 1, 0))
    print('# points in 2 valid sets, in 0 valid regions, in 1 train region: ' + str(vs2_vr0_tr1))
    vs2_vr0_tr2 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==0, num_train_in_region==2)), 1, 0))
    print('# points in 2 valid sets, in 0 valid regions, in 2 train regions: ' + str(vs2_vr0_tr2))
    vs2_vr1_tr0 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==1, num_train_in_region==0)), 1, 0))
    print('# points in 2 valid sets, in 1 valid region, in 0 train regions: ' + str(vs2_vr1_tr0))
    vs2_vr1_tr1 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==1, num_train_in_region==1)), 1, 0))
    print('# points in 2 valid sets, in 1 valid region, in 1 train region: ' + str(vs2_vr1_tr1))
    vs2_vr1_tr2 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==1, num_train_in_region==2)), 1, 0))
    print('# points in 2 valid sets, in 1 valid region, in 2 train regions: ' + str(vs2_vr1_tr2))
    vs2_vr2_tr0 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==2, num_train_in_region==0)), 1, 0))
    print('# points in 2 valid sets, in 2 valid regions, in 0 train regions: ' + str(vs2_vr2_tr0))
    vs2_vr2_tr1 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==2, num_train_in_region==1)), 1, 0))
    print('# points in 2 valid sets, in 2 valid regions, in 1 train region: ' + str(vs2_vr2_tr1))
    vs2_vr2_tr2 = np.sum(np.where(np.logical_and.reduce((num_valid==2, num_valid_in_region==2, num_train_in_region==2)), 1, 0))
    print('# points in 2 valid sets, in 2 valid regions, in 2 train regions: ' + str(vs2_vr2_tr2))

def evaluate_test_region_fold_consistency(fold_results_dict, X, d, t):
    '''
    Evaluate consistency of grouping of agents with data in the region across folds

    Parameters
    ==========
    folds_results_dict: dictionary
        Contains keys 0, 1, 2, 3 mapped to the results dictionary for that fold.
        Results dictionary contains region model, region indices, outcome predictions, agent groupings, etc.
        
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
    ''' 
    indicator_idxs = np.empty((len(fold_results_dict[0]['test_idxs']),4))
    for fold_idx in range(4):
        indicator_idxs[:,fold_idx] \
            = np.where(np.isin(np.arange(indicator_idxs.shape[0]),fold_results_dict[fold_idx]['test_region_idxs']), 1, 0)
    point_in_region_times = np.sum(indicator_idxs, axis=1)
    for i in range(5):
        print('# test points that appear in region ' + str(i) + ' times: ' \
              + str(np.sum(np.where(point_in_region_times==i, 1, 0))))

def evaluate_fold_consistency(fold_results_dict, X, d, t):
    '''
    Evaluate consistency of whether a point is in the region when it is part of a validation vs training set

    Parameters
    ==========
    folds_results_dict: dictionary
        Contains keys 0, 1, 2, 3 mapped to the results dictionary for that fold.
        Results dictionary contains region model, region indices, outcome predictions, agent groupings, etc.
        
    X: numpy array
        Contains context of samples
        
    d: numpy array
        Contains agents
        
    t: numpy array
        Contains treatment decisions
    ''' 
    indicator_idxs = np.empty((X.shape[0],4))
    n_prov = int(np.max(d)+1)
    fold_region_idxs = dict()
    for fold_idx in range(4):
        train_region_idxs = fold_results_dict[fold_idx]['train_idxs'][fold_results_dict[fold_idx]['train_region_idxs']]
        valid_region_idxs = fold_results_dict[fold_idx]['valid_idxs'][fold_results_dict[fold_idx]['valid_region_idxs']]
        test_region_idxs = fold_results_dict[fold_idx]['test_idxs'][fold_results_dict[fold_idx]['test_region_idxs']]
        all_region_idxs = np.concatenate((train_region_idxs, valid_region_idxs, test_region_idxs))
        fold_region_idxs[fold_idx] = all_region_idxs
        indicator_idxs[:,fold_idx] = np.where(np.isin(np.arange(X.shape[0]),all_region_idxs), 1, 0)
    point_in_region_times = np.sum(indicator_idxs, axis=1)
    num_points_in_regions = np.zeros(5)
    for i in range(5):
        num_points_in_regions[i] = np.sum(np.where(point_in_region_times == i, 1, 0))
    region_agreement = np.sum(np.maximum(4 - np.sum(indicator_idxs, axis=1), 
                                         np.sum(indicator_idxs, axis=1)))/float((4*X.shape[0]))
    at_least_present_once_idxs = np.nonzero(np.where(np.sum(indicator_idxs, axis=1) >= 1, 1, 0))[0]
    at_least_present_once_indicator_idxs = indicator_idxs[at_least_present_once_idxs]
    region_agreement_selected \
        = np.sum(np.maximum(4 - np.sum(at_least_present_once_indicator_idxs, axis=1), 
                            np.sum(at_least_present_once_indicator_idxs, axis=1)))/float((4*len(at_least_present_once_idxs)))
    print('Region agreement across 4 folds: ' + str(region_agreement))
    print('Region agreement among points selected in at least 1 fold: ' + str(region_agreement_selected))
    for i in range(len(num_points_in_regions)):
        print('# points that appear in region ' + str(i) + ' times: ' + str(num_points_in_regions[i]))
    if fold_results_dict[0]['pred_provider_split'] is None:
        return
    provider_in_region_idxs = np.empty((n_prov,4))
    for fold_idx in range(4):
        d_in_region = np.unique(fold_results_dict[fold_idx]['d'][fold_region_idxs[fold_idx]])
        provider_in_region_idxs[:,fold_idx] \
            = np.where(np.isin(np.arange(len(fold_results_dict[fold_idx]['pred_provider_split'])), d_in_region), 1, 0)
    pair_agreement = 0
    num_pairs_in_2regions = 0
    num_pairs_in_region = 0
    num_pairs_always_same_side = 0
    num_pairs_always_diff_side = 0
    num_pairs_3folds_same_side = 0
    num_pairs_3folds_diff_side = 0
    num_pairs_in_3regions = 0
    num_pairs_in_4regions = 0
    provider_in_region_times = np.sum(provider_in_region_idxs, axis=1)
    num_providers_in_regions = np.zeros(5)
    for i in range(5):
        num_providers_in_regions[i] = np.sum(np.where(provider_in_region_times == i, 1, 0))
    for i in range(n_prov):
        for j in range(i, n_prov):
            pair_in_region_idxs = np.sum(provider_in_region_idxs[[i,j]], axis=0)
            region_contains_pair  = np.where(pair_in_region_idxs == 2, 1, 0)
            num_times_pair_in_region = np.sum(region_contains_pair)
            if num_times_pair_in_region >= 1:
                num_pairs_in_region += 1
            if num_times_pair_in_region >= 2:
                num_pairs_in_2regions += 1
                num_same_group = 0
                num_diff_group = 0
                always_same = True
                always_diff = True
                for fold_idx in range(4):
                    if region_contains_pair[fold_idx] == 1:
                        if fold_results_dict[fold_idx]['pred_provider_split'][i] \
                            == fold_results_dict[fold_idx]['pred_provider_split'][j]:
                            num_same_group += 1
                            always_diff = False
                        else:
                            num_diff_group += 1
                            always_same = False
                pair_agreement += max(num_same_group, num_diff_group)/float(num_times_pair_in_region)
                if always_same:
                    num_pairs_always_same_side += 1
                if always_diff:
                    num_pairs_always_diff_side += 1
                if num_same_group >= 3:
                    num_pairs_3folds_same_side += 1
                if num_diff_group >= 3:
                    num_pairs_3folds_diff_side += 1
                if num_same_group + num_diff_group >= 3:
                    num_pairs_in_3regions += 1
                if num_same_group + num_diff_group == 4:
                    num_pairs_in_4regions += 1
    if num_pairs_in_2regions == 0:
        pair_agreement = 0
    else:
        pair_agreement /= float(num_pairs_in_2regions)
    print('Partition agreement across 4 folds: ' + str(pair_agreement))
    print('# pairs in >=2 regions: ' + str(num_pairs_in_2regions))
    print('# pairs in any region: ' + str(num_pairs_in_region))
    for i in range(len(num_providers_in_regions)):
        print('# providers that appear in region ' + str(i) + ' times: ' + str(num_providers_in_regions[i]))
    print('# pairs always same side: ' + str(num_pairs_always_same_side))
    print('# pairs always diff side: ' + str(num_pairs_always_diff_side))
    print('num_pairs_3folds_same_side: ' + str(num_pairs_3folds_same_side))
    print('num_pairs_3folds_diff_side: ' + str(num_pairs_3folds_diff_side))
    print('num_pairs_in_3regions: ' + str(num_pairs_in_3regions))
    print('num_pairs_in_4regions: ' + str(num_pairs_in_4regions))
    
