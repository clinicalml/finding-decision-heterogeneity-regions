#!/usr/bin/env python
# coding: utf-8

DATA_DIR = # PATH TO RECIDIVISM DATA
DEFAULT_CACHE_PATH = # PATH TO CACHE

import pandas as pd
import numpy as np
from scipy.special import logit, expit
import pickle as pkl

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
from sklearn import model_selection as ms
from sklearn import pipeline

import semisynth_subsets as subset

def split_semi_synthetic_data(X_syn, d_syn, t_syn, n_providers, true_provider_split,
        verbose=True):
    '''
    Split semi-synthetic dataset into train, validate, and test sets. Filters out agents
    with fewer than 3 samples.

    Parameters
    ----------
    X_syn : array-like of shape (n_samples, n_features)
        Feature matrix.

    d_syn : array-like of shape (n_samples,)
        Decision-makers.

    t_syn : array-like of shape (n_samples,)
        Binary decisions.

    n_providers : int
        Number of providers present in d_syn.

    true_provider_split : array-like of shape (n_providers,)
        True provider groupings.
    
    Returns
    -------
    X_syn : array-like of shape (n_samples_new, n_features)
        Feature matrix, possibly smaller than the input matrix,
        i.e. n_samples_new <= n_samples.

    d_syn : array-like of shape (n_samples_new,)
        Decision-makers/agents corresponding to X_syn.

    t_syn : array-like of shape (n_samples_new,)
        Decisions corresponding to X_syn.

    train_idxs : array-like of shape (n_train_samples,)
        Array containing indices for the training set.

    valid_idxs : array-like of shape (n_valid_samples,)
        Array containing indices for the valid set.

    test_idxs : array-like of shape (n_test_samples,)
        Array containing indices for the test set.

    true_provider_split : array-like of shape (n_providers_new,)
        Provider groupings, after filtering out providers with
        fewer than 3 samples.
    '''

    providers_to_remove = []
    indices_to_keep = np.ones(X_syn.shape[0])
    providers_to_keep = np.ones(n_providers)
    for i in range(n_providers):
        prov_idxs = np.nonzero(np.where(d_syn == i, 1, 0))[0]
        # if provider has fewer than 3 samples, remove provider
        if len(prov_idxs) < 3:
            providers_to_remove.append(i)
            providers_to_keep[i] = 0
            for j in prov_idxs:
                indices_to_keep[j] = 0
    assert np.sum(indices_to_keep) > 0 # not all samples removed
    assert np.sum(providers_to_keep) >= 2

    indices_to_keep = np.nonzero(np.where(indices_to_keep == 1, 1, 0))[0]
    X_syn = X_syn[indices_to_keep]
    d_syn = d_syn[indices_to_keep]
    t_syn = t_syn[indices_to_keep]
    providers_to_keep = np.nonzero(np.where(providers_to_keep == 1, 1, 0))[0]
    true_provider_split = true_provider_split[providers_to_keep]
    for provider in providers_to_remove[::-1]:
        d_syn = np.where(d_syn > provider, d_syn-1, d_syn)
    n_providers -= len(providers_to_remove)
    assert len(np.unique(d_syn)) == n_providers
    assert np.min(d_syn) == 0
    assert np.max(d_syn) == n_providers - 1
    assert n_providers >= 2 # at least 2 providers remain
    if verbose:
        print(str(n_providers) + ' providers in samples')

    for i in range(n_providers):
        prov_idxs = np.nonzero(np.where(d_syn == i, 1, 0))[0]

        rng = np.random.default_rng(seed=972)
        rng.shuffle(prov_idxs)
        prov_train_idxs = prov_idxs[:min(int(.6*len(prov_idxs)),len(prov_idxs)-2)]
        prov_valid_idxs = prov_idxs[min(int(.6*len(prov_idxs)),len(prov_idxs)-2):int(.8*len(prov_idxs))] # guarantee >= 1 sample
        prov_test_idxs = prov_idxs[int(.8*len(prov_idxs)):] # int() guarantees at most len(prov_idxs)-1, so >= 1 sample
        if i == 0:
            train_idxs = prov_train_idxs
            valid_idxs = prov_valid_idxs
            test_idxs = prov_test_idxs
        else:
            train_idxs = np.concatenate((train_idxs, prov_train_idxs))
            valid_idxs = np.concatenate((valid_idxs, prov_valid_idxs))
            test_idxs = np.concatenate((test_idxs, prov_test_idxs))
    return X_syn, d_syn, t_syn, train_idxs, valid_idxs, test_idxs, true_provider_split

def load_semi_synthetic_compas_data(
        seed=0,
        num_agents=None,
        logit_adjust=0.5,
        subset_func=subset.drug_possession,
        drop_rare_charges=False,
        verbose=False,
        cache=False,
        cache_path=DEFAULT_CACHE_PATH):

    if cache:
        cache_file = f'{cache_path}/semisynth_cache.pkl'
        raise NotImplementedError("Caching not yet implemented")

    # Import the cleaned data, and collect feature / label names
    df = pd.read_csv(f'{DATA_DIR}/05-19-1800-compas_no_feedback_data.csv')

    label_name = 'outcome'
    col_names = [f for f in df.columns.values 
                      if label_name not in f
                      and '_id' not in f
                 ]

    user_feat_names = [f for f in col_names if 'user_' in f or 'time_deciding' in f]
    case_feat_names = [f for f in col_names if f not in user_feat_names]

    # Remove race as a feature, as it was not used in the actual user study
    case_feat_names = [f for f in case_feat_names if 'race' not in f]

    if drop_rare_charges:

        # To simplify, split out charge / non-charge features
        # Only keep the top 10 charge features (plus whether misdemeanor / felony)
        charge_feats = [f for f in case_feat_names if 'charge' in f]
        no_charge_feats = [f for f in case_feat_names if f not in charge_feats]

        top_charge_feats = pd.DataFrame.from_dict(
            # Get the mean of each charge feature
            {f: df[f].mean() for f in charge_feats}, orient='index', columns=['freq']
            # Sort in descending order by frequency
            ).sort_values(by='freq', ascending=False
            # Get top 11 (charge_degree + top 10)
            ).head(11).index.values.tolist()

        if verbose:
            lowest_freq = pd.DataFrame.from_dict(
                {f: df[f].mean() for f in top_charge_feats}, orient='index', columns=['freq']
                ).sort_values(by='freq', ascending=False
                ).min()[0]
            print(f"Filtered out low-frequency charges, lowest frequency is {lowest_freq:.3f}")

        case_feat_names = no_charge_feats + top_charge_feats

    if verbose:
        print("Features:")
        print(case_feat_names)

    X = df[case_feat_names].copy()
    y = df[label_name].copy()

    '''
    Choosing a subpopulation.

    In this script, we currently chose individuals with drug-related offenses

    You can change this by defining a different subset algorithm (see above)
    '''
    LOGIT_ADJUST = logit_adjust
    X, subset_idx, true_region_func = subset_func(X)

    true_beta = subset_idx.mean()

    if verbose:
        print(f"True beta: {true_beta:.1%}")

    '''
    Estimating the average policy

    We use logistic regression to estimate the average policy
    of the actual users across the entire population:
    * One group of synthetic agents will apply this policy to all individuals
    * Another group of synthetic agents will apply a modified policy (see below)
    '''

    lr = pipeline.Pipeline(
                [('scaler', preprocessing.StandardScaler()),
                 ('logReg', LogisticRegression(
                     fit_intercept=True,
                     solver = 'lbfgs',
                     random_state=0, max_iter=1000))])

    lr_params = {
        'logReg__C': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    }

    lr_cv = ms.GridSearchCV(lr, scoring='roc_auc', param_grid=lr_params)

    lr_cv.fit(X, y)
    preds = lr_cv.predict_proba(X)
    logits = np.log(preds[:, 1] / preds[:, 0])

    if verbose:
        print(f"Average Policy best parameters {lr_cv.best_params_}")

    '''
    Modifying the alternative policy

    Because we have access to the logits of the baseline policy,
    we can perform a simple adjustment to simulate from both policies
    * For the baseline, we sample from the predictive dist. of the logits
    * For the alternative, we add to the "coefficients" of the subset,
      by increasing the logits by a fixed amount, then transforming back 
      to probabilities and sampling
    '''

    # We need to ensure arrays have the correct (1, -1) shape
    if len(logits.shape) == 1:
        logits = logits.reshape(-1, 1)

    subset_array = subset_idx.values.reshape(-1, 1)

    # Before we can modify logits, we need to create and assign synthetic
    # agents

    # Get the number of agents 
    if num_agents is None:
        num_agents = df['user_id'].nunique()

    # Randomly assign agents to individuals 
    rng = np.random.default_rng(seed=seed)
    agent_id = rng.choice(num_agents, size = (X.shape[0], 1))

    # Split agents into two groups based on ID
    # Note >= to account for zero-indexing.  If num_agents = 2, need this to
    # assign to two separate groups
    agent_group = np.where(agent_id >= int(num_agents / 2), 1, 0)

    # We collect all of the agent info (by sample) into a dataframe, including 
    # * the original logits, 
    # * whether or not this sample falls in the subset, 
    # * and so on.

    # Collect the original data
    A = pd.DataFrame(np.hstack((agent_id, agent_group, logits, subset_array)),
                     columns = ['agent_id', 'agent_group',
                                'orig_logit', 'subset_idx'])
    A = A.astype({'agent_id': int, 'agent_group': int})

    # Create a bool idx for rows to modify [sample in subset + agent in grp. 1]
    modify_idx = np.logical_and(subset_idx, A['agent_group'] == 1)

    # Modify logits
    A['new_logit'] = A['orig_logit'].copy()
    A.loc[modify_idx, 'new_logit'] += LOGIT_ADJUST

    # Record the original and new probabilities, for reference
    A['orig_p'] = expit(A['orig_logit'])
    A['new_p'] = expit(A['new_logit'])

    # Track the divergence in probabilities
    group_1_idx = A['agent_group'] == 1
    alt_subset = np.logical_and(group_1_idx, subset_idx)
    base_subset = np.logical_and(~group_1_idx, subset_idx)

    alt_avg_prob = A.loc[alt_subset].mean()['new_p']
    base_avg_prob = A.loc[base_subset].mean()['new_p']
    if verbose:
        print(f"Base policy on subset: {base_avg_prob}\n"
              f"Alt policy on subset: {alt_avg_prob}")

    '''
    Examine the impact, get descriptive stats

    These are commented out for now, but can be used to get estimates of
    various quantities of how our random generation worked.
    '''
    # # Recall: Bool index for subset is subset_idx
    # group_1_idx = A['agent_group'] == 1
    # # modify_idx is the AND of subset_idx and group_1_idx
    #
    # tex_df = pd.DataFrame(columns = ['Original Prob.', 'Modified Prob.'], 
    #              index = ['Overall', 
    #                       'Base Policy', 
    #                       'Alt. Policy', 
    #                       'Subset',
    #                       'Base Policy + Subset',
    #                       'Alt. Policy + Subset',
    #                       'Base Policy + Complement',
    #                       'Alt. Policy + Complement',
    #                      ])
    #
    # tex_df.loc['Overall', :] = \
    #         A.mean()[['orig_p', 'new_p']].values
    # tex_df.loc['Base Policy', :] = \
    #         A.loc[~group_1_idx].mean()[['orig_p', 'new_p']].values
    # tex_df.loc['Alt. Policy', :] = \
    #         A.loc[group_1_idx].mean()[['orig_p', 'new_p']].values
    # tex_df.loc['Subset', :] = \
    #         A.loc[subset_idx].mean()[['orig_p', 'new_p']].values
    #
    # tex_df.loc['Alt. Policy + Subset', :] = \
    #         A.loc[np.logical_and(group_1_idx, subset_idx)
    #                 ].mean()[['orig_p', 'new_p']].values
    #
    # tex_df.loc['Base Policy + Subset', :] = \
    #         A.loc[np.logical_and(~group_1_idx, subset_idx)
    #                 ].mean()[['orig_p', 'new_p']].values
    #
    # tex_df.loc['Alt. Policy + Complement', :] = \
    #         A.loc[np.logical_and(group_1_idx, ~subset_idx)
    #                 ].mean()[['orig_p', 'new_p']].values
    #
    # tex_df.loc['Base Policy + Complement', :] = \
    #         A.loc[np.logical_and(~group_1_idx, ~subset_idx)
    #                 ].mean()[['orig_p', 'new_p']].values
    #
    # print(tex_df.to_latex(float_format="{:0.2%}".format))

    '''
    Consolidate final output

    This produces the outputs required by the synthetic experiment scripts
    '''

    A = A[['agent_id', 'agent_group', 'subset_idx', 'new_p']]
    A = A.rename(columns = {'subset_idx': 'subset',
                            'new_p': 'decision_prob'})

    # X needs to be in numpy format
    out_X = X.values
    out_d = A['agent_id'].values

    # Generate binary decisions
    rng = np.random.default_rng(seed=seed)
    out_t = rng.binomial(p=A['decision_prob'], n=1)

    true_provider_split = A['agent_group'].values
    n_providers = len(np.unique(A['agent_id'].values))

    # Redudant check, but can never be too safe
    assert np.array_equal(true_region_func(out_X), (A['subset'] == 1).values),\
            "true_region_func does not match the given region!"

    out_X, out_d, out_t, \
        train_idxs, valid_idxs, test_idxs, \
        true_provider_split = \
        split_semi_synthetic_data(
                out_X, out_d, out_t,
                n_providers, true_provider_split,
                verbose=False)

    data_dict = {}
    data_dict['alt_avg_prob'] = alt_avg_prob
    data_dict['base_avg_prob'] = base_avg_prob
    data_dict['X_pd'] = X.copy()
    data_dict['feat_names'] = X.columns.values.tolist()
    data_dict['X'] = out_X
    data_dict['d'] = out_d
    data_dict['t'] = out_t
    data_dict['train_idxs'] = train_idxs
    data_dict['valid_idxs'] = valid_idxs
    data_dict['test_idxs'] = test_idxs
    data_dict['true_provider_split'] = true_provider_split
    data_dict['true_region_func'] = true_region_func
    data_dict['true_beta'] = true_beta

    return data_dict
