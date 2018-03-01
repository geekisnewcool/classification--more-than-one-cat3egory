#predicting the foresttype(kaggle)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('G:/Sopranos/ml/Tree')


#impoting the datasets
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#creating independent and dependent variables

y=train.iloc[:,55]
x=train.iloc[:,1:55]

for i in range(len(train)):
    if x['Vertical_Distance_To_Hydrology'][i]<0:
        x['Vertical_Distance_To_Hydrology'][i]=0
    
for j in range(len(test)):
   if test['Vertical_Distance_To_Hydrology'][j]<0:
       test['Vertical_Distance_To_Hydrology'][j]=0 
        

x['Distance_to_water_body']=((x.Vertical_Distance_To_Hydrology)**2+(x.Horizontal_Distance_To_Hydrology)**2)**0.5
x=x.drop(['Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology'],axis=1)
test['Distance_to_water_body']=((test.Vertical_Distance_To_Hydrology)**2+(test.Horizontal_Distance_To_Hydrology)**2)**0.5
test=test.drop(['Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology'],axis=1)
 
x['Average_Hillshade']=((x.Hillshade_9am)+(x.Hillshade_Noon)+(x.Hillshade_3pm))/3
x=x.drop(['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'],axis=1)
test['Average_Hillshade']=((test.Hillshade_9am)+(test.Hillshade_Noon)+(test.Hillshade_3pm))/3
test=test.drop(['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'],axis=1)
test_final=test.iloc[:,1:52]
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)

# =============================================================================
# from sklearn.cross_validation import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier=classifier.fit(x,y)

y_pred=classifier.predict(test_final)
answer=pd.DataFrame(y_pred)
answer.to_csv('G:/Sopranos/ml/Tree/answer.csv')

