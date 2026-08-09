from sklearn.tree import DecisionTreeClassifier
import numpy as np

X=np.array([[25,18],
             [2,45],
             [14,19],
             [6,27]

])
y=np.array([1,0,1,0])

model=DecisionTreeClassifier()
model.fit(X,y)
V=[[10,46]]
prediction=model.predict(V)
if prediction[0]==1:
    print("Likely")
else:
    print("Unlikely")
