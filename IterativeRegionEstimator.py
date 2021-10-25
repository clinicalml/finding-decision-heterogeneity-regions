import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor

class IterativeRegionEstimator(BaseEstimator):
    '''
    Identifies a region of disagreement using the IterativeAlg method.

    Parameters
    ----------
    region_modelclass : BaseEstimator, default=RandomForestRegressor()
        An unfitted model class that has a .fit(X, y) method and a .predict(X) method.

    beta : float, default=0.25
        A real number between 0 and 1 representing the size of the desired region.

    n_iter : int, default=10
        Maximum number of iterations of the algorithm.

    Attributes
    ----------
    grouping_ : dictionary
        A dictionary mapping each agent to a binary grouping.

    region_model_ : BaseEstimator
        A fitted model of the same class as self.region_modelclass.

    threshold_ : float
        Defines the identified region of variation as the inputs x such that 
        region_model.predict(X) >= threshold.

    '''

    def __init__(self, region_modelclass=RandomForestRegressor(), beta=0.25, n_iter=10):
        self.region_modelclass = region_modelclass
        self.beta = beta
        self.n_iter = n_iter

        # This is passed into .fit()
        self.outcome_model_ = None

    def _best_grouping(self, S, X, y, a, preds):
        '''
        Identifies the best grouping given a region.

        Parameters
        ----------
        S : array-like of shape (n_samples,)
            A list of booleans indicating membership in the current region.

        X, y, a : data inherited from .fit().

        preds : array-like of shape (n_samples,)
            A list of floats representing predictions of the outcome_model passed into .fit().

        Returns
        -------
        G : dictionary
            A dictionary mapping each unique element of a to a binary grouping.

        q_score : float
            The value hat{Q}(S, G), a measure of the variation on S under grouping G.
        '''

        assert S.dtype == np.dtype('bool')

        # Put everyone in group 0
        G = {}
        for agent in np.unique(a):
            G[agent] = 0

        # Put agents with positive total residual on S into group 1
        q_score = 0.0
        for agent in np.unique(a):
            ixs = (a[S] == agent)
            if np.sum(ixs) > 0:
                term = (1/np.sum(S)) * np.sum(y[S][ixs] - preds[S][ixs])
                if term >= 0:
                    G[agent] = 1
                    q_score += term

        return G, q_score

    def _best_region(self, G, X, y, a, preds):
        '''
        Identifies the best region given a grouping.

        Parameters
        ----------
        G : dictionary
            A dictionary mapping each unique element of a to a binary grouping.

        X, y, a : data inherited from .fit().

        preds : array-like of shape (n_samples,)
            A list of floats representing predictions of the outcome_model passed into .fit().

        Returns
        -------
        region_model : BaseEstimator
            A fitted estimator of the same class as self.region_modelclass.
        '''
        
        # Get the groupings for agents of each data point
        g = np.zeros(len(a))
        for i in range(len(a)):
            g[i] = G[a[i]]

        # Train model to predict residuals in group 1
        res = (y - preds) * g
        region_model = self.region_modelclass.fit(X, res)
        
        return region_model

    def fit(self, X, y, a, outcome_model):
        '''
        Fits the estimator to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        a : array-like of shape (n_samples,)
            Agent labels relative to X.

        outcome_model
            A fitted predictor with a .predict_proba(X) method such that 
            .predict_proba(X)[:, 1] consists of real numbers between 0 and 1.

        Returns
        -------
        self
            Fitted estimator.
        '''

        # Get predictions from outcome_model
        preds = outcome_model.predict_proba(X)[:, 1]

        # Store the outcome model
        self.outcome_model_ = outcome_model

        # Initialize S to the entire space
        S = np.array([True] * X.shape[0])
        G = None
        G_prev = None
        region_model = None
        threshold = None

        for it in range(self.n_iter):
            # Find the best grouping for the current region
            G, q_score = self._best_grouping(S, X, y, a, preds)
            if G_prev is not None and G_prev == G:
                break
            G_prev = G

            # Find the best region for the current grouping
            region_model = self._best_region(G, X, y, a, preds)
            region_scores = region_model.predict(X)
            threshold = np.quantile(region_scores, 1-self.beta)

            # NOTE: If this is >= for decision trees, then the region will tend
            # to be larger than (perhaps) desired, but it will avoid problems
            # (for small beta) of not selecting any group at all.
            S = region_scores >= threshold

            # Alternative logic
            # S = region_scores > threshold
            # if np.sum(S) == 0:
            #     S = region_scores >= threshold

        G, q_score = self._best_grouping(S, X, y, a, preds)

        # Store fitted model attributes
        self.grouping_ = G
        self.region_model_ = region_model
        self.threshold_ = threshold

        return self

    def predict(self, X):
        '''
        Classifies data points inside/outside the region.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            A list of booleans indicating membership in the identified region.

        '''
        region_scores = self.region_model_.predict(X)
        predictions = region_scores >= self.threshold_

        # Implement alternative logic, strict thresholding
        # predictions = region_scores > self.threshold_
        # if np.sum(predictions) == 0:
        #     predictions = region_scores >= self.threshold_

        return predictions

    def score(self, X, y, a, preds=None):
        '''
        Generate a Q-score, normalized by the size of the inferred region.

        First, this predicts region membership in X, and then computes the
        score, using the fitted outcome_model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        a : array-like of shape (n_samples,)
            Agent labels relative to X.

        preds : array-like of shape (n_samples,)
            Optional, if provided these will be used in place of the
            outcome_model predictions

        Returns
        -------
        q_score : float
            The value hat{Q}(S, G), a measure of the variation
            on S (determined by self.region_model) under grouping G
            (determined by taking all agents with positive average scores)
        '''

        S = self.predict(X)
        if preds is None:
            preds = self.outcome_model_.predict_proba(X)[:, 1]

        _, q_score = self._best_grouping(S, X, y, a, preds)

        return q_score


