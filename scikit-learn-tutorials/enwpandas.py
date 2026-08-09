import pandas as pd
#Get the data
car_sales=pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-extended.csv")
#Transform door column to object type
car_sales['Doors']=car_sales['Doors'].astype(object)
category=['Make','Colour','Doors']
#bring in the get_dummies method
dummy=pd.get_dummies(data=car_sales[category],dtype=float)
#Train_test_split
y=car_sales['Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    dummy,y,test_size=0.2,random_state=42
)
#Import model
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(random_state=42)
#Train our model
model.fit(X_train,y_train)
#Evaluate the model
score=model.score(X_test,y_test)
print(score)