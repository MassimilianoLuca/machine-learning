from numpy import genfromtxt
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import learning_curve

# used only to produce images for the report
import matplotlib.pyplot as plt

X_train = genfromtxt('train-data.csv', delimiter=',')
y_train = genfromtxt('train-targets.csv', dtype=int, delimiter=',')
X_test = genfromtxt('test-data.csv', delimiter=',')
kf = KFold(n_splits=3, shuffle=True, random_state=42)
gamma_values = [0.1, 0.02, 0.01, 0.001]
accuracy_scores = []
f1_scores = []
recall_scores = []
precision_scores = []

for gamma in gamma_values:

    clf = svm.SVC(C=10, kernel='rbf', gamma=gamma)

    acc_score = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='accuracy')
    f1p_score = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='f1')
    rec_score = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='recall')
    prec_score = cross_val_score(clf, X_train, y_train, cv=kf.split(X_train), scoring='precision')

    f1_score = f1p_score.mean()
    recall_score = rec_score.mean()
    precision_score = prec_score.mean()
    accuracy_score = acc_score.mean()

    f1_scores.append(f1_score)
    recall_scores.append(recall_score)
    precision_scores.append(precision_score)
    accuracy_scores.append(accuracy_score)

best_index = np.array(accuracy_scores).argmax()
best_gamma = gamma_values[best_index]
#################################################
# Purpose: assignment report only               #
#################################################
#print('best-gamma')
#print(best_gamma)
#print('ACC')
#print(accuracy_scores[best_index])
#print('F1')
#print(f1_scores[best_index])
#print('REC')
#print(recall_scores[best_index])
#print('PRE')
#print(precision_scores[best_index])
#################################################
clf = svm.SVC(C=10, kernel='rbf', gamma=best_gamma)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

np.savetxt('test-targets.txt', y_pred, fmt='%d')

plt.figure()
plt.title("Learning curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.legend()
plt.savefig('plot.png', dpi=300)
