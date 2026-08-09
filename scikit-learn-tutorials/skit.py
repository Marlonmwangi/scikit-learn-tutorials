import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
print(f"Using scikit learn version{sklearn.__version__}")
from sklearn.model_selection import train_test_split
from sklearn import datasets

# heart_disease = pd.read_csv("../data/heart-disease.csv") # load data from local directory
heart_disease = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv") # load data directly from URL (requires raw form on GitHub, source: https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/data/heart-disease.csv)
heart_disease.head()
X=heart_disease.drop("target",axis=1)
y=heart_disease["target"]

#Divide the data
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.25
)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
model_acc=round(model.score(X_test,y_test)*100,2)
print(model_acc)




