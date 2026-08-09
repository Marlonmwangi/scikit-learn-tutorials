import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
#INITIALIZE OUR DATASET
brst=load_breast_cancer()
df=pd.DataFrame(brst.data,columns=brst.feature_names)
#INITIALIZE FEATURES AND TARGET
df["target"]=brst.target
#INITIALIZE X and y
X=df.drop("target",axis=1)
y=df["target"]
#IMPORT AND INITIALIZE OUR MODEL
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
param_grid={'n_estimators':[i for i in range(100,200,10)]}
#Gridsearch
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    verbose=1
)
#fit the grid
grid.fit(X,y)
#Display our best estimator value for efficient cross validation
print(f"The best estimator is {grid.best_params_}")
print(f"It had a score of {round(grid.best_score_*100,2)}%")
