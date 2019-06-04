from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

H_source = loadmat('C:/Users/hancylv/Desktop/research/CDH/H.mat')
H_df = pd.DataFrame(H_source['H'])

X_train = H_df.iloc[:, :-1]
X_train = preprocessing.scale(X_train)
Y_train = H_df.iloc[:, 229]

# clf_SVM = SVC(C=1.38, cache_size=200, class_weight=None, coef0=0.0,
#               decision_function_shape='ovr', degree=3, gamma=0.0048, kernel='rbf',
#               max_iter=-1, probability=True, random_state=51, shrinking=True,
#               tol=0.001, verbose=False)
# # clf_SVM.fit(X_train, Y_train)
# scores = cross_val_score(clf_SVM, X_train, Y_train, cv=6, scoring='accuracy')
# print('SVM', scores.mean())  # 0.9234781976828196

# clf_LG = LogisticRegression(C=1.36, random_state=51, solver='liblinear')
# scores_LG = cross_val_score(clf_LG, X_train, Y_train, cv=6)
# print('LG', scores_LG.mean())  # 0.9242527151826833
#
# clf_Bayes = GaussianNB()  # 0.8558497785031159
# # clf_Bayes.fit(X_train, Y_train)
# scores_Bayes = cross_val_score(clf_Bayes, X_train, Y_train, cv=6)
# print('Bayes', scores_Bayes.mean())
#
# clf_KNN = KNeighborsClassifier(n_neighbors=10, n_jobs=3)  # 0.786864779036046
# # clf_KNN.fit(X_train, Y_train)
# scores_KNN = cross_val_score(clf_KNN, X_train, Y_train, cv=6)
# print('KNN', scores_KNN.mean())
#
#
# clf_DT = DecisionTreeClassifier(criterion='entropy')  # gini 0.8044503446087589 entropy 0.8094684428879845
# # clf_DT.fit(X_train, Y_train)
# scores_DT = cross_val_score(clf_DT, X_train, Y_train, cv=6)
# print('DT', scores_DT.mean())
#
# clf_RF = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=2)
# # clf_RF.fit(X_train, Y_train)
# scores_RF = cross_val_score(clf_RF, X_train, Y_train, cv=6)
# print('RF', scores_RF.mean())
#
# clf_MLP = MLPClassifier(hidden_layer_sizes=(400,), activation='logistic')  # 0.901838584121105
# # clf_MLP.fit(X_train, Y_train)
# scores_MLP = cross_val_score(clf_MLP, X_train, Y_train, cv=6)
# print('MLP', scores_MLP.mean())  # 0.9203888752345174 200 logistic 0.9205813273447573 1000 identity
# 0.9221287919642839 1000 logistic
#
# clf_bag = BaggingClassifier(
#                             n_estimators=10, max_samples=1, max_features=1,
#                             bootstrap=True, bootstrap_features=False, oob_score=False,
#                             warm_start=False, n_jobs=-1, random_state=51, verbose=0)
# scores_bag = cross_val_score(clf_bag, X_train, Y_train, cv=6)  # mlp0.9031906809534825
# print('bag', scores_bag.mean())
#
# clf_Ad = AdaBoostClassifier(n_estimators=500,
#                             learning_rate=0.5,
#                             algorithm='SAMME.R', random_state=51)
# clf_Ad.fit(X_train, Y_train)
# scores_Ad = cross_val_score(clf_Ad, X_train, Y_train, cv=6)
# print('AD', scores_Ad.mean())  # 0.9109217256528584
#
# clf_Vote = VotingClassifier(estimators=[('rf', clf_RF), ('svm', clf_SVM), ('mlp', clf_MLP)],
#                             n_jobs=-1, voting='soft')
# clf_Vote.fit(X_train, Y_train)
# scores_Vote = cross_val_score(clf_Vote, X_train, Y_train, cv=6)
# print('Vote', scores_Vote.mean())

# SVM 0.8766753248358633
# LG 0.8779862282401742
# Bayes 0.7782767249664563
# KNN 0.8139236896976807
# DT 0.7340987368652362
# RF 0.8283524970745882
# MLP 0.8738336487219137
# bag 0.4970479588645736
# AD 0.8709971415540049
# Vote 0.8834565910367065

RANDOM_SEED = 51
np.random.seed(RANDOM_SEED)

clf_SVM = SVC(C=1.38, gamma=0.0048, probability=True, random_state=RANDOM_SEED, kernel='rbf')
# clf_RF = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=4, random_state=RANDOM_SEED)
clf_RF = RandomForestClassifier(n_estimators=490, max_depth=13, min_samples_split=2,
                                min_samples_leaf=2, oob_score=True,
                                random_state=RANDOM_SEED, max_features=9, n_jobs=4)
clf_MLP = MLPClassifier(hidden_layer_sizes=(400,), solver='adam',
                        learning_rate='constant', learning_rate_init=0.01,
                        activation='logistic', random_state=RANDOM_SEED)
# clf_LG = LogisticRegression(C=1.36, solver='liblinear', random_state=RANDOM_SEED, penalty='l2')
# clf_Bayes = GaussianNB()  # 0.8558497785031159
clf_Ad = AdaBoostClassifier(n_estimators=500,
                            learning_rate=0.5,
                            algorithm='SAMME.R', random_state=RANDOM_SEED)
clf_Vote = VotingClassifier(estimators=[('Ad', clf_Ad), ('svm', clf_SVM), ('mlp', clf_MLP), ('RF', clf_RF)],
                            n_jobs=-1, voting='soft')
clf_Vote.fit(X_train, Y_train)
scores_Vote = cross_val_score(clf_Vote, X_train, Y_train, cv=6)
print(scores_Vote.mean())
