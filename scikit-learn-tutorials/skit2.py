import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 

hd=pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
X=hd.drop("target",axis=1)
y=hd["target"]
#MODEL
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42)
#Split training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.20,random_state=42
)
#Train your model
model.fit(X_train,y_train)
#Evaluate your model &predict with your model
from sklearn.metrics import accuracy_score,precision_score
y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)
print(f"Accuracy score:{acc:.2f}")
prec=precision_score(y_test,y_pred)
print(f"Precision:{prec:.2f}")
#Evaluate training accuracy vs testing accuracy
train_acc=model.score(X_train,y_train)
test_acc=model.score(X_test,y_test)
print(f"Train Accuracy:{train_acc:.2f}")
print(f"Test  Accuracy:{test_acc:.2f}")
