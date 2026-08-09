import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#READ YOUR DATA
df = pd.DataFrame({
    'Neighborhood': ['Downtown', 'Suburbs', 'Downtown'],
    'Condition': ['Good', 'Poor', 'Fair'],
    'Square_Feet': [1500, 2200, 1100]
})
#start
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#BUILD THE COLUMN TRANSFORMER
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
preprocessor=ColumnTransformer(
    transformers=[
        ('geo_encode', OneHotEncoder(), ['Neighborhood']),
        ('state_encoder',OrdinalEncoder(),['Condition']),
    ],
    remainder='passthrough'
)
#MODEL
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
#PIPELINE
from sklearn.pipeline import Pipeline
pipeline=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('model',model)
    ]
)
processed=preprocessor.fit_transform(df)
print(f"Processed Data: {processed}")


