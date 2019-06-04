import pyswarms as ps
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

C_source = loadmat('C:/Users/hancylv/Desktop/research/CDH/C.mat')
C_df = pd.DataFrame(C_source['C'])

X_train = C_df.iloc[:, :-1]
X_train = preprocessing.StandardScaler().fit_transform(X_train)
Y_train = C_df.iloc[:, 229]

clf_SVM = SVC(C=1.33, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=0.0035, kernel='linear',
              max_iter=-1, probability=True, random_state=51, shrinking=True,
              tol=0.001, verbose=False)

clf_MLP = MLPClassifier(hidden_layer_sizes=(200,), activation='identity', random_state=51)

clf_Ad = AdaBoostClassifier(n_estimators=500,
                            learning_rate=0.5,
                            algorithm='SAMME.R', random_state=51)


def aimfuc(k):
    obj = np.ones(k.shape[0])
    for i in range(0, k.shape[0]):
        vote = VotingClassifier(estimators=[('ad', clf_Ad), ('svm', clf_SVM), ('mlp', clf_MLP)],
                                n_jobs=6, voting='soft', weights=[k[i, 0], k[i, 1], k[i, 2]])
        scores_vote = cross_val_score(vote, X_train, Y_train, cv=6, n_jobs=-1)
        f = scores_vote.mean()
        obj[i] = -f
    return obj


# Set-up hyperparameters
options = {'c1': 1.49445, 'c2': 1.49445, 'w': 0.9}

bounds = (np.array([1, 1, 1]), np.array([9, 9, 9]))
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=2, dimensions=3, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(aimfuc, iters=1)
