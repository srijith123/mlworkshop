#FINDING APPLES OR ORANGES
from sklearn.tree import DecisionTreeClassifier
features=[[140,0],[130,0],[150,1],[170,1]] #0-smooth 1-Bumpy
labels=[0,0,1,1] #0-apple 1-orange


clf=DecisionTreeClassifier()
clf.fit(features,labels)

pre=clf.predict([[160,1]])
print("Prediction = ",pre) 
