import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

def evaluate_with_bootstrap(encodings, labels, n_bootstraps=1000, alpha=0.05):
    """
    Evaluates encoding quality using bootstrapping to estimate the R^2 score distribution.
    """
    n_samples = len(encodings)
    bootstrap_scores = []
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = encodings[indices], labels[indices]
        
        # Out-of-bag samples for validation
        oob_indices = np.array(list(set(range(n_samples)) - set(indices)))
        if len(oob_indices) == 0:
            continue 
            
        X_oob, y_oob = encodings[oob_indices], labels[oob_indices]

        regressor = Ridge()
        regressor.fit(X_boot, y_boot)
        score = regressor.score(X_oob, y_oob)
        bootstrap_scores.append(score)
        
    lower_bound = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, lower_bound, upper_bound

def evaluate_with_cv(encodings, labels, n_splits=5, n_repeats=20, alpha=0.05):
    """
    Evaluates encoding quality using Repeated K-Fold Cross-Validation.
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    cv_scores = []

    for train_idx, test_idx in cv.split(encodings):
        X_train, X_test = encodings[train_idx], encodings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        regressor = Ridge()
        regressor.fit(X_train, y_train)

        score = regressor.score(X_test, y_test)
        cv_scores.append(score)

    mean_score = np.mean(cv_scores)
    lower_bound = np.percentile(cv_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(cv_scores, 100 * (1 - alpha / 2))

    return mean_score, lower_bound, upper_bound