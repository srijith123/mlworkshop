#Using knn Classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier() 
from sklearn.datasets import load_iris

iris=load_iris()

features=iris.data
labels=iris.target

#splitting to training and testing(70-30)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(features, labels, test_size=.3)  #.3 means it splits into 70-30


knn.fit(X_train, Y_train)  

#Predict

p=knn.predict(X_test)

#TO Determine the accuracy of our model

from sklearn.metrics import accuracy_score


print("Accuracy= ",accuracy_score(Y_test,p))


