#IMPORT TOOLS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#FETCH DATASET OR WRITE YOUR OWN DATASET
from sklearn.datasets import load_breast_cancer
#LOAD DATASET
brst=load_breast_cancer()
#PANDAS DATAFRAME
df=pd.DataFrame(brst.data,columns=brst.feature_names)
df["target"]=brst.target
#INITIALIZE FEATURES AND TARGET
X=df.drop("target",axis=1)
y=df["target"]
'''
#PREPROCESSING
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(
    Transformer=[
        ()
    ],
    remainder='passthrough'
)

#PIPELINE
from sklearn.pipeline import Pipeline
pipeline=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('model',model)
    ]
)
#SPLIT
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)'''
#MODEL IMPORT AND INITIALIZATION
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42)
#TRAIN AND TEST MODEL
from sklearn.model_selection import GridSearchCV
#CROSS VALIDATION
param_grid={'n_estimators':[i for i in range(100,200,10)]}
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,verbose=1,scoring='recall')
#EVALUATE MODEL
grid.fit(X,y)
#PRINT RESULTS
print(f"Best estimator is {grid.best_params_} and its score is {round(grid.best_score_*100,2)}%")