import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from itertools import product
import matplotlib.pyplot as plt

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv("C:\\Users\\User\\Desktop\\Papers\\diabetes.csv")

array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

# create the sub models
estimators = [] #creating estimators array for appending all the outputs from different classifier

# model1 is based on Logistic Regression classifier
model1 = LogisticRegression()
estimators.append(('logistic', model1))
results1 = model_selection.cross_val_score(model1, X, Y, cv=kfold)
accuracy1=results1.mean()

plt.figure()
plt.scatter([0,1,2,3,4,5,6,7,8,9],results1,c='k',label='Results')
plt.xlabel('Fold No.')
plt.ylabel('Accuracy')
plt.title('Logistic Regression       Accuracy = ' + str(accuracy1*100))
plt.show()

# model2 is based on Random forest classifier
model2 = RandomForestClassifier()
estimators.append(('rf', model2))
results2 = model_selection.cross_val_score(model2, X, Y, cv=kfold)
accuracy2=results2.mean()

plt.figure()
plt.scatter([0,1,2,3,4,5,6,7,8,9],results2,c='k',label='Results')
plt.xlabel('Fold No.')
plt.ylabel('Accuracy')
plt.title('Random Forest       Accuracy = ' + str(accuracy2*100))
plt.show()


# model3 is based on support vector classifier
model3 = SVC()
estimators.append(('svm', model2))
results3 = model_selection.cross_val_score(model3, X, Y, cv=kfold)
accuracy3=results3.mean()

plt.figure()
plt.scatter([0,1,2,3,4,5,6,7,8,9],results3,c='k',label='Results')
plt.xlabel('Fold No.')
plt.ylabel('Accuracy')
plt.title('Support Vector Machine       Accuracy = ' + str(accuracy3*100))
plt.show()



# model4 is based on Gaussian Naive Bayes
model4= GaussianNB()
estimators.append(('gnb', model4))
results4 = model_selection.cross_val_score(model4, X, Y, cv=kfold)
accuracy4=results4.mean()

plt.figure()
plt.scatter([0,1,2,3,4,5,6,7,8,9],results4,c='k',label='Results')
plt.xlabel('Fold No.')
plt.ylabel('Accuracy')
plt.title('Naive Bayes      Accuracy = ' + str(accuracy4*100))
plt.show()



# create the ensemble model
ensemble = VotingClassifier(estimators,voting='soft', weights=[1, 2, 1, 1])
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)

accuracy=results.mean()
#print(results.mean())
print("Accuracy: %.2f%%" % (accuracy * 100.0))

plt.figure()
plt.scatter([0,1,2,3,4,5,6,7,8,9],results,c='k',label='Results')
plt.plot(color='navy')
plt.xlabel('Fold No.')
plt.ylabel('Accuracy')
plt.title('Soft Voting       Accuracy = ' + str(accuracy*100))
plt.show()

