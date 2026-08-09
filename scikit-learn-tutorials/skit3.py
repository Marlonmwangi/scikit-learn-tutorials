#IMPORT KEY PYTHON LIBRARIES
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
#Get your dataset
from sklearn.datasets import load_breast_cancer
#Initialize your dataset
brst=load_breast_cancer()
#MAKE THE DATASET INTO A PANDAS DATAFRAME
df=pd.DataFrame(brst.data,columns=brst.feature_names)
#INITIALIZE YOUR FEATURES AND TARGET
df["target"]=brst.target
X=df.drop("target",axis=1)
y=df["target"]
#Split your data into training and testing batches
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.20,random_state=42
)
#CHOOSE YOUR MODEL AND INITIALIZE IT
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42)
#TRAIN YOUR MODEL
model.fit(X_train,y_train)
#EVALUATE YOUR MODEL
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
y_pred=model.predict(X_test)
model_test_acc=round(model.score(X_test,y_test)*100,2)
model_acc=round(accuracy_score(y_test,y_pred)*100,2)
model_prec=round(precision_score(y_test,y_pred)*100,2)
model_rec=round(recall_score(y_test,y_pred)*100,2)
#VIEW YOUR EVALUATION
print(model_test_acc,model_acc,model_prec,model_rec)
print(classification_report(y_test,y_pred))