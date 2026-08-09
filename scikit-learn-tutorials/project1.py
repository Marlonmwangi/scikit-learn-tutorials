from sklearn.tree import DecisionTreeClassifier
import numpy as np

#FEATURES:[Transaction_Amount_KES, Time_Of_Day_24hr, Failed_Login_Attempts]
T=np.array([
    [10000,13,6],
    [9000,6,1],
    [21000,14,11],
    [19000,23,2],
    [5000,7,8]
])
w=np.array([1,0,1,0,1])
model=DecisionTreeClassifier()
model.fit(T,w)
X=np.array([3000,9,6])
prediction=model.predict([X])
if prediction[0]==1:
    print("Likely Fraud!")
else:
    print("Legitimate")

