#IMPORT ENCODER AND TRANSFORMER
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#EXTRACT DATA 
import pandas as pd
Od=pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
#CREATE AN INSTANCE FOR OUR ENCODER
encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
#CREATE AN INSTANCE FOR OUR TRANSFORMER
transformer=ColumnTransformer(
    transformers=[
        ('encoder',encoder,['Cabin'])
    ],
    remainder='passthrough'
)
#Fit and transform the data
transformed=transformer.fit_transform(Od)
transformed=pd.DataFrame(transformed,columns=transformer.get_feature_names_out())
print(transformed)
