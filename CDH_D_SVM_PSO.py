from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import pyswarms as ps

D_source = loadmat('C:/Users/hancylv/Desktop/research/CDH/D.mat')
D_df = pd.DataFrame(D_source['D'])

X_train = D_df.iloc[:, :-1]
X_train = preprocessing.StandardScaler().fit_transform(X_train)
Y_train = D_df.iloc[:, 229]


def aimfuc(k):
    obj = np.ones(k.shape[0])
    for i in range(0, k.shape[0]):
        clf_svm = SVC(C=k[i, 0], cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovr', degree=3, gamma=k[i, 1], kernel='rbf',
                      max_iter=-1, probability=False, random_state=51, shrinking=True,
                      tol=0.001, verbose=False)
        scores_svm = cross_val_score(clf_svm, X_train, Y_train, cv=6, n_jobs=-1)
        f = scores_svm.mean()
        obj[i] = -f
    return obj


# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# init_pos = np.array([1.25, 0.0035])
bounds = (np.array([0.1, 0.001]), np.array([2, 2]))
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(aimfuc, iters=1)

