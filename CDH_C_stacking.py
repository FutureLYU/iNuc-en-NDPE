# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
#     GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
import numpy as np
# from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
import pandas as pd

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer

'''创建训练的数据集'''
C_source = loadmat('C:/Users/hancylv/Desktop/research/CDH/C.mat')
C_df = pd.DataFrame(C_source['C'])
X_C = C_df.iloc[:, :-1]
X_C = preprocessing.StandardScaler().fit_transform(X_C)
Y_C = C_df.iloc[:, 229]
Y_C = np.array(Y_C)


def mcc(esmitator, x_ori, y_real):
    y_pred = esmitator.predict(x_ori)
    mat = confusion_matrix(y_real, y_pred)
    tn = mat[0, 0]
    fn = mat[1, 0]
    tp = mat[1, 1]
    fp = mat[0, 1]
    mcc_return = ((tp+tn)/(tp+tn+fp+fn))/(((1+((fp+fn)/(tp+fn)))*(1+((fn+fp)/(tn+fp))))**0.5)
    return mcc_return


'''模型融合中使用到的各个单模型'''
# clfs = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
#         RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
#         ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
#         ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
#         GradientBoostingClassifier(learning_rate=0.05, subsample=0.5,
#                                    max_depth=6, n_estimators=5)]

# clfs = [SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True)]

scorer = {'mcc': mcc, 'accuracy': make_scorer(accuracy_score),
          'precision': make_scorer(precision_score),
          'recall': make_scorer(recall_score)}

RANDOM_SEED = 51


clf_SVM = SVC(C=1.25, gamma=0.0035, probability=True, random_state=RANDOM_SEED, kernel='linear')
clf_RF = RandomForestClassifier(n_estimators=290, max_depth=9, min_samples_leaf=10, max_features='sqrt',
                                min_samples_split=30, n_jobs=6,
                                oob_score=True, random_state=10)

# clf_RF = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=2, random_state=RANDOM_SEED)
clf_MLP = MLPClassifier(hidden_layer_sizes=(500,), learning_rate='constant',
                        solver='adam', learning_rate_init=0.001,
                        activation='logistic', random_state=RANDOM_SEED)
clf_LG = LogisticRegression(C=1.51, solver='liblinear', random_state=RANDOM_SEED, penalty='l2')
# clf_Bayes = GaussianNB()  # 0.8558497785031159
clf_Ad = AdaBoostClassifier(n_estimators=500,
                            learning_rate=0.5,
                            algorithm='SAMME.R', random_state=RANDOM_SEED)
# clf_KNN = KNeighborsClassifier(n_neighbors=10, n_jobs=3)


sclf = StackingCVClassifier(classifiers=[clf_SVM, clf_RF, clf_MLP, clf_Ad],
                            meta_classifier=clf_LG, use_probas=True)
# sclf.fit(X_C, Y_C)
scores_ST = cross_validate(sclf, X_C, Y_C, cv=6, scoring=scorer, n_jobs=3)
print('accuracy', scores_ST['test_accuracy'].mean())
print('precision', scores_ST['test_precision'].mean())
print('recall', scores_ST['test_recall'].mean())
print('mcc', scores_ST['test_mcc'].mean())
