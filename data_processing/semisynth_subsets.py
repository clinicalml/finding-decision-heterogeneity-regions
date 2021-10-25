import pandas as pd
import numpy as np

'''
Subset Selection Options

You can define a new function here (and provide to the loading function)
which defines a subset on which to increase the logits!

Your new function should return:
X: DataFrame, possibly with fewer features
subset_idx: Boolean array indicating membership in the subset
true_region_func: Function which takes a NUMPY array of the same shape as X,
and returns a boolean array (should match subset_idx)

**Note on subset definitions**:
You can define an arbitrary subset here, and the code will work.

However, for interpretation, note that the code effectively adds a new
indicator variable with logistic regression coefficient of LOGIT_ADJUST,
to construct the alternative policy.

To ensure that the resulting alternative policy is still in the class of
logistic regression models on the original features, you may wish to choose
a subset that already corresponds to an indicator variable in the data.
'''

def misdemeanor_under35(X):
    '''
    Misdemeanors, age less than or equal to 35 years old

    Around 21% of the dataset
    '''
    # Misdemeanor
    mis_str = 'charge_degree (misd/fel)'
    age_str = 'age'
    mis_idx = np.squeeze(np.where(X.columns.values == mis_str))
    age_idx = np.squeeze(np.where(X.columns.values == age_str))

    subset_idx = np.logical_and(X[mis_str] == 0, X[age_str] <= 35)

    # Define and evaluate true region function
    def true_region_func(X):
        return np.logical_and(X[:, mis_idx] == 0, X[:, age_idx] <= 35)

    assert np.array_equal(true_region_func(X.values), subset_idx)

    return X, subset_idx, true_region_func


def drug_possession(X):
    '''
    Drug possession charge, based on matching 'possession' in the feature,
    including:
    ['charge_possession_of_cocaine',
     'charge_possession_of_a_controlled_substance',
     'charge_possession_of_cannabis/marijuana',
     'charge_possession_of_ecstasy',
     'charge_possession_of_morphine',
     'charge_possession_of_meth',
     'charge_possession_of_oxycodone',
     'charge_possession_of_lsd',
     'charge_possession_of_heroin']

    Around 22% of the dataset if low-freq charges are not removed first.
    '''
    # All drug-posssesion related charges (from one-hot encoding of charge)
    possession_feats = [f for f in X.columns.values if 'possession' in f]
    subset_idx = X[possession_feats].sum(axis=1) == 1

    column_idxs = np.where(
            np.array(['possession' in f for f in X.columns.values]))[0]

    # Define and evaluate true region function
    def true_region_func(X):
        return X[:, column_idxs].sum(axis=1) == 1

    assert np.array_equal(true_region_func(X.values), subset_idx)

    return X, subset_idx, true_region_func