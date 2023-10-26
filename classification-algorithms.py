import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_report', 'safety', 'class_values']
cars = pd.read_csv("car.data", header=None, names=col_names)

le = LabelEncoder()
cars['buying'] = le.fit_transform(cars['buying'])
cars['maint'] = le.fit_transform(cars['maint'])
cars['doors'] = le.fit_transform(cars['doors'])
cars['persons'] = le.fit_transform(cars['persons'])
cars['lug_report'] = le.fit_transform(cars['lug_report'])
cars['safety'] = le.fit_transform(cars['safety'])
onehotencoder = OneHotEncoder()

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
data = np.array(columnTransformer.fit_transform(cars), dtype=np.str)

feature_col_names = ['buying', 'maint', 'doors', 'persons', 'lug_report', 'safety']
X = cars[feature_col_names]
y = cars.class_values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#########################    DECISION TREE     #########################
# Create Decision Tree classifier object
dtree = DecisionTreeClassifier(criterion="gini", max_depth=3)
# Train Decision Tree Classifier
dtree = dtree.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = dtree.predict(X_test)

print("Decision Tree:")

# Model Accuracy, how often is the classifier correct?
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: {0:0.2f}".format(accuracy))

# Model Precision
precision = metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("Precision: {0:0.2f}".format(precision))

# Model Recall
recall = metrics.recall_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("Recall: {0:0.2f}".format(recall))

# Model F1
f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("F1: {0:0.2f}".format(f1))
print("\n")

# plot the decision tree
tree.plot_tree(dtree, filled=True)
plt.show()
######################### END OF DECISION TREE #########################

######################### K-NEAREST NEIGHBOURS #########################
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("K-Nearest Neighbours:")
# Model Accuracy, how often is the classifier correct?
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: {0:0.2f}".format(accuracy))

# Model Precision
precision = metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("Precision: {0:0.2f}".format(precision))

# Model Recall
recall = metrics.recall_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("Recall: {0:0.2f}".format(recall))

# Model F1
f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("F1: {0:0.2f}".format(f1))
print("\n")
######################### END OF K-NEAREST NEIGHBOURS #########################

#########################        NAIVE BAYES          #########################
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Naive Bayes:")
# Model Accuracy, how often is the classifier correct?
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: {0:0.2f}".format(accuracy))

# Model Precision
precision = metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("Precision: {0:0.2f}".format(precision))

# Model Recall
recall = metrics.recall_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("Recall: {0:0.2f}".format(recall))

# Model F1
f1 = metrics.f1_score(y_test, y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
print("F1: {0:0.2f}".format(f1))
print("\n")
#########################     END OF NAIVE BAYES      #########################