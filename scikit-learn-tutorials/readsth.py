from sklearn.datasets import load_breast_cancer
import pandas as pd
brst=load_breast_cancer()
df=pd.DataFrame(brst.data,columns=brst.feature_names)
print(df.head())