import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
#IMPORT YOUR DATASET
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = (bank_marketing.data.targets['y']=='yes').astype(int)
#IMPORT TRAIN_TEST_SPLIT
from sklearn.model_selection import train_test_split
# After loading X, y — encode categoricals
X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42
)
#IMPORT MODEL
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42,n_estimators=1000)
#Train our model
model.fit(X_train,y_train)
#IMPORT ROC
from sklearn.metrics import roc_curve,roc_auc_score
y_prob=model.predict_proba(X_test)[:, 1]
#Calculate tpr and fpr
fpr, tpr, thresholds=roc_curve(y_test,y_prob)
#calculate Area under ROC curve
auc_score=roc_auc_score(y_test,y_prob)
print(f"Area under ROC curve ={auc_score}")

#PLOT THE ROC CURVE
plt.figure(figsize=(8,6))

plt.plot(fpr,tpr,color='blue',lw=2,label=f'ROC curve(AUC={auc_score:.2f})')
#PLOT A DASHED RANDOM WEIGHT BASELINE
plt.plot([0,1],[0,1],ls='--',color='red',label='random guess')

#format the plot
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('fpr,(1-specificity)')
plt.ylabel('tpr,(sensitivity)')
plt.title('ROC CURVE DISPLAY')
plt.legend()
plt.grid(True,ls=':',alpha=0.6)

plt.show()
