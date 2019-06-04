import numpy as np
import geatpy as ga

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


def aimfuc(Phen, legV):
    k1 = Phen[:, [0]]
    k2 = Phen[:, [1]]
    k3 = Phen[:, [2]]
    obj = np.array(k1)
    for i in range(0, k1.shape[0]):
        vote = VotingClassifier(estimators=[('ad', clf_Ad), ('svm', clf_SVM), ('mlp', clf_MLP)],
                                n_jobs=4, voting='soft', weights=[k1[i, 0], k2[i, 0], k3[i, 0]])
        scores_vote = cross_val_score(vote, X_train, Y_train, cv=6, n_jobs=4)
        f = scores_vote.mean()
        obj[i, 0] = f
    return [obj, legV]


AIM_M = __import__('CDH_C_GA')
x1 = [1, 9]
x2 = [1, 9]
x3 = [1, 9]
b1 = [1, 1]
b2 = [1, 1]
b3 = [1, 1]
codes = [0, 0, 0]
precisions = [4, 4, 4]
scales = [0, 0, 0]
ranges = np.vstack([x1, x2, x3]).T
borders = np.vstack([b1, b2, b3]).T

FieldD = ga.crtfld(ranges, borders, precisions, codes, scales)

[pop_trace, var_trace, times] = ga.sga_new_code_templet(AIM_M, 'aimfuc', None, None, FieldD,
                                                        problem='R', maxormin=-1, MAXGEN=1,
                                                        NIND=2, SUBPOP=1, GGAP=0.8, selectStyle='sus',
                                                        recombinStyle='xovdp', recopt=None, pm=None,
                                                        distribute=True, drawing=1)

