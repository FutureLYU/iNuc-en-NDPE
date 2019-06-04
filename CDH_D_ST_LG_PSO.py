
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pyswarms as ps
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier

C_source = loadmat('C:/Users/hancylv/Desktop/research/CDH/D.mat')
C_df = pd.DataFrame(C_source['D'])

X_train = C_df.iloc[:, :-1]
X_train = preprocessing.StandardScaler().fit_transform(X_train)
Y_train = C_df.iloc[:, 229]
Y_train = np.array(Y_train)

RANDOM_SEED = 51
np.random.seed(RANDOM_SEED)
clf_SVM = SVC(C=1.76, gamma=0.00525, probability=True, random_state=RANDOM_SEED, kernel='rbf')
# clf_RF = RandomForestClassifier(n_estimators=130, max_depth=13, min_samples_split=2,
#                                 min_samples_leaf=2, max_features=9, oob_score=True,
#                                 random_state=RANDOM_SEED, n_jobs=4)
clf_RF = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1, random_state=RANDOM_SEED)
clf_MLP = MLPClassifier(hidden_layer_sizes=(500,), learning_rate='constant', learning_rate_init=0.01,
                        activation='logistic', random_state=RANDOM_SEED)
clf_Ad = AdaBoostClassifier(n_estimators=500,
                            learning_rate=0.5,
                            algorithm='SAMME.R', random_state=RANDOM_SEED)


def aimfuc(k):

    obj = np.ones(k.shape[0])

    for i in range(0, k.shape[0]):

        clf_LG = LogisticRegression(C=k[i, 0], solver='liblinear', random_state=RANDOM_SEED, penalty='l2')
        sclf = StackingCVClassifier(classifiers=[clf_SVM, clf_RF, clf_MLP, clf_Ad],
                                    meta_classifier=clf_LG, use_probas=True, cv=4)

        # sclf.fit(X_C, Y_C)
        scores_ST = cross_val_score(sclf, X_train, Y_train, cv=6, scoring='accuracy', n_jobs=-1)
        f = scores_ST.mean()  # 0.9001018065904675 3svm
        obj[i] = -f
    return obj


# Set-up hyperparameters
options = {'c1': 1.49445, 'c2': 1.49445, 'w': 0.9}

bounds = (np.array([0.1]), np.array([2]))
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(aimfuc, iters=60)
