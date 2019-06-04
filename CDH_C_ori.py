from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import make_scorer


def mcc(esmitator, x_ori, y_real):
    y_pred = esmitator.predict(x_ori)
    mat = confusion_matrix(y_real, y_pred)
    tn = mat[0, 0]
    fn = mat[1, 0]
    tp = mat[1, 1]
    fp = mat[0, 1]
    mcc_return = ((tp+tn)/(tp+tn+fp+fn))/(((1+((fp+fn)/(tp+fn)))*(1+((fn+fp)/(tn+fp))))**0.5)
    return mcc_return


C_source = loadmat('C:/Users/hancylv/Desktop/research/CDH/C.mat')
C_df = pd.DataFrame(C_source['C'])

X_train = C_df.iloc[:, :-1]
X_train = preprocessing.StandardScaler().fit_transform(X_train)

# X_train = preprocessing.normalize(X_train)
Y_train = C_df.iloc[:, 229]
Y_train = np.array(Y_train)

scorer = {'mcc': mcc, 'accuracy': make_scorer(accuracy_score),
          'precision': make_scorer(precision_score),
          'recall': make_scorer(recall_score)}

clf_SVM = SVC(probability=True, gamma='scale')
# clf_SVM.fit(X_train, Y_train)
scores_SVM = cross_validate(clf_SVM, X_train, Y_train, cv=6, scoring=scorer)


clf_LG = LogisticRegression()
scores_LG = cross_validate(clf_LG, X_train, Y_train, cv=6, scoring=scorer)


clf_Bayes = GaussianNB()  # 0.8558497785031159
# clf_Bayes.fit(X_train, Y_train)
scores_Bayes = cross_validate(clf_Bayes, X_train, Y_train, cv=6, scoring=scorer)


clf_KNN = KNeighborsClassifier()  # 0.786864779036046
# clf_KNN.fit(X_train, Y_train)
scores_KNN = cross_validate(clf_KNN, X_train, Y_train, cv=6, scoring=scorer)


clf_DT = DecisionTreeClassifier()  # gini 0.8044503446087589 entropy 0.8094684428879845
# clf_DT.fit(X_train, Y_train)
scores_DT = cross_validate(clf_DT, X_train, Y_train, cv=6, scoring=scorer)


clf_RF = RandomForestClassifier()
# clf_RF.fit(X_train, Y_train)
scores_RF = cross_validate(clf_RF, X_train, Y_train, cv=6, scoring=scorer)

clf_MLP = MLPClassifier(hidden_layer_sizes=(200,), activation='logistic')  # 0.901838584121105
# clf_MLP.fit(X_train, Y_train)
scores_MLP = cross_validate(clf_MLP, X_train, Y_train, cv=6, scoring=scorer)

clf_bag = BaggingClassifier()
scores_bag = cross_validate(clf_bag, X_train, Y_train, cv=6, scoring=scorer)

clf_Ad = AdaBoostClassifier()
# clf_Ad.fit(X_train, Y_train)
scores_Ad = cross_validate(clf_Ad, X_train, Y_train, cv=6, scoring=scorer)
# RANDOM_SEED = 51
# np.random.seed(RANDOM_SEED)

# clf_SVM = SVC(C=1.38, gamma=0.0048, probability=True, random_state=RANDOM_SEED, kernel='rbf')
# # clf_RF = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=4, random_state=RANDOM_SEED)
# clf_RF = RandomForestClassifier(n_estimators=490, max_depth=13, min_samples_split=2,
#                                 min_samples_leaf=2, oob_score=True,
#                                 random_state=RANDOM_SEED, max_features=9, n_jobs=4)
# clf_MLP = MLPClassifier(hidden_layer_sizes=(400,), solver='adam',
#                         learning_rate='constant', learning_rate_init=0.01,
#                         activation='logistic', random_state=RANDOM_SEED)
# clf_LG = LogisticRegression(C=1.36, solver='liblinear', random_state=RANDOM_SEED, penalty='l2')
# # clf_Bayes = GaussianNB()  # 0.8558497785031159
# clf_Ad = AdaBoostClassifier(n_estimators=500,
#                             learning_rate=0.5,
#                             algorithm='SAMME.R', random_state=RANDOM_SEED)
clf_Vote = VotingClassifier(estimators=[('Ad', clf_Ad), ('svm', clf_SVM), ('mlp', clf_MLP), ('RF', clf_RF)],
                            n_jobs=-1, voting='soft')

scores_Vote = cross_validate(clf_Vote, X_train, Y_train, cv=6, scoring=scorer)

