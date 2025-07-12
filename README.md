# iris-flower-classification-1
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("IRIS.csv")

df.head()

df.info()

df.describe()

sns.pairplot(df, hue="species")
plt.show()

X = df.drop("species", axis=1)
X

Y = df["species"]
Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)

pred1 = model_svc.predict(X_test)
print("Accuracy:",accuracy_score(y_test, pred1)*100)

from sklearn.linear_model import LogisticRegression 
Lr = LogisticRegression()
Lr.fit(X_train, y_train)

pred2 = Lr.predict(X_test)
print("Accuracry:",accuracy_score(y_test, pred2)*100)

print("Classificaton Report:",classification_report(y_test, pred2))

# prompt: outcome of the this project overview about 

print("This project is likely a classification task on the Iris dataset. The goal is to build and compare different machine learning models (Support Vector Machine and Logistic Regression) to predict the species of an Iris flower based on its measurements (sepal length, sepal width, petal length, petal width). The performance of the models will be evaluated using metrics such as accuracy and classification report."
