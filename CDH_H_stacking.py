import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer,confusion_matrix

'''创建训练的数据集'''
C_source = loadmat('C:/Users/hancylv/Desktop/research/CDH/H.mat')
C_df = pd.DataFrame(C_source['H'])
X_C = C_df.iloc[:, :-1]
X_C = preprocessing.StandardScaler().fit_transform(X_C)
Y_C = C_df.iloc[:, 229]
Y_C = np.array(Y_C)

'''模型融合中使用到的各个单模型'''


def mcc(esmitator, x_ori, y_real):
    y_pred = esmitator.predict(x_ori)
    mat = confusion_matrix(y_real, y_pred)
    tn = mat[0, 0]
    fn = mat[1, 0]
    tp = mat[1, 1]
    fp = mat[0, 1]
    mcc_return = ((tp+tn)/(tp+tn+fp+fn))/(((1+((fp+fn)/(tp+fn)))*(1+((fn+fp)/(tn+fp))))**0.5)
    return mcc_return


scorer = {'mcc': mcc, 'accuracy': make_scorer(accuracy_score),
          'precision': make_scorer(precision_score),
          'recall': make_scorer(recall_score)}

RANDOM_SEED = 51
np.random.seed(RANDOM_SEED)

clf_SVM = SVC(C=1.38, gamma=0.0048, probability=True, random_state=RANDOM_SEED, kernel='rbf')
# clf_RF = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=4, random_state=RANDOM_SEED)
clf_RF = RandomForestClassifier(n_estimators=490, max_depth=13, min_samples_split=2,
                                min_samples_leaf=2, oob_score=True,
                                random_state=RANDOM_SEED, max_features=9, n_jobs=4)
clf_MLP = MLPClassifier(hidden_layer_sizes=(400,), solver='adam', learning_rate='constant', learning_rate_init=0.01,
                        activation='logistic', random_state=RANDOM_SEED)
clf_LG = LogisticRegression(C=1.36, solver='liblinear', random_state=RANDOM_SEED, penalty='l2')
# clf_Bayes = GaussianNB()  # 0.8558497785031159
clf_Ad = AdaBoostClassifier(n_estimators=500,
                            learning_rate=0.5,
                            algorithm='SAMME.R', random_state=RANDOM_SEED)
# clf_KNN = KNeighborsClassifier(n_neighbors=10, n_jobs=3)

# clfs = [SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True),
#         SVC(C=1.25, gamma=0.0035, probability=True)]

sclf = StackingCVClassifier(classifiers=[clf_SVM, clf_RF, clf_MLP, clf_Ad],
                            meta_classifier=clf_SVM, use_probas=True)
# sclf.fit(X_C, Y_C)
scores_ST = cross_validate(sclf, X_C, Y_C, cv=6, scoring=scorer, n_jobs=4)  # 0.8871348822966395
print('accuracy', scores_ST['test_accuracy'].mean())
print('precision', scores_ST['test_precision'].mean())
print('recall', scores_ST['test_recall'].mean())
print('mcc', scores_ST['test_mcc'].mean())

