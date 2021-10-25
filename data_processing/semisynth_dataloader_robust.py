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

from semisynth_dataloader import split_semi_synthetic_data
import semisynth_subsets as subset

def load_semi_synthetic_compas_data(
        seed=0,
        num_agents=None,
        num_groups=2,
        logit_adjust=[0,0.5],
        subset_func=subset.drug_possession,
        drop_rare_charges=False,
        verbose=False,
        cache=False,
        cache_path=DEFAULT_CACHE_PATH):
    
    assert num_agents >= num_groups
    assert len(logit_adjust) == num_groups

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
    #LOGIT_ADJUST = logit_adjust
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

    # Split agents into groups based on ID
    agent_group = np.mod(agent_id, num_groups)

    # We collect all of the agent info (by sample) into a dataframe, including 
    # * the original logits, 
    # * whether or not this sample falls in the subset, 
    # * and so on.

    # Collect the original data
    A = pd.DataFrame(np.hstack((agent_id, agent_group, logits, subset_array)),
                     columns = ['agent_id', 'agent_group',
                                'orig_logit', 'subset_idx'])
    A = A.astype({'agent_id': int, 'agent_group': int})

    # Modify logits
    A['new_logit'] = A['orig_logit'].copy()
    for group_idx in range(num_groups):
        group_subset_idxs = np.logical_and(subset_idx, A['agent_group'] == group_idx)
        A.loc[group_subset_idxs,'new_logit'] = A.loc[group_subset_idxs,'orig_logit'] + logit_adjust[group_idx]
    
    # Record the original and new probabilities, for reference
    A['orig_p'] = expit(A['orig_logit'])
    A['new_p'] = expit(A['new_logit'])

    # Track the probability for each group
    group_avg_probs = np.zeros(num_groups)
    for group_idx in range(num_groups):
        group_subset_idxs = np.logical_and(subset_idx, A['agent_group'] == group_idx)
        group_avg_probs[group_idx] = A.loc[group_subset_idxs,'new_p'].mean()
        if verbose:
            print('Average policy for group ' + str(group_idx) + ': ' + str(group_avg_probs[group_idx]))
        
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
    data_dict['group_avg_probs'] = group_avg_probs
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
