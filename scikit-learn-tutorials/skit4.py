import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn 
from sklearn.datasets import load_breast_cancer 
#INITIALIZE YOUR DATASET
brst=load_breast_cancer()
df=pd.DataFrame(brst.data,columns=brst.feature_names)
#INITIALIZE YOUR FEATURES AND TARGET
df["target"]=brst.target
#Initialize X and y 
X=df.drop("target",axis=1)
y=df["target"]
#IMPORT THE MODEL
from sklearn.ensemble import RandomForestClassifier
#Initialize your model
model=RandomForestClassifier(random_state=42)
np.random_seed=42
param_grid={'n_estimators':[i for i in range(100,200,10)]}
#Perform a grid search
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    verbose=1
)
#fit the grid to the data
grid.fit(X,y)
#Display our best estimator value for efficient cross validation
print(f"The best estimator is {grid.best_params_}")
print(f"It had a score of {round(grid.best_score_*100,2)}%")
