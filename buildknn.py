from scipy.spatial import distance

def eucli(a,b):
	return distance.euclidean(a,b)
class myknn():
	#fit Method
	def fit(self,X_train, Y_train):
		self.X_train=X_train
		self.Y_train=Y_train
	#Predict Method
	def predict(self,X_test):
		predictions=[]
		for row in X_test:
			labels = self.closest(row)
			predictions.append(labels)
		return predictions
	#Closest Distacnce
	def closest(self,row):
		best_dist = eucli(row,self.X_train[0])
		base_index = 0
		for i in range(1,len(self.X_train)):
			dist=eucli(row, self.X_train[i])
			if(dist < best_dist):
				best_dist=dist
				base_index=i
		return  self.Y_train[base_index]

from sklearn.neighbors import KNeighborsClassifier
knn = myknn() 
from sklearn.datasets import load_iris

iris=load_iris()

x=iris.data
y=iris.target

#splitting to training and testing(70-30)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(x, y, test_size=.3)  #.3 means it splits into 70-30


knn.fit(X_train, Y_train)  

#Predict

p=knn.predict(X_test)

#TO Determine the accuracy of our model

from sklearn.metrics import accuracy_score


print("Accuracy= ",accuracy_score(Y_test,p))


