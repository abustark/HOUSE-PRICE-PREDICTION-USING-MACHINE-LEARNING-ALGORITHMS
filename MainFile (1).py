#======================= IMPORT PACKAGES ============================

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#======================= DATA SELECTION =========================

print("=======================================")
print("---------- Data Selection -------------")
print("=======================================")
data=pd.read_csv('Bengaluru_House_Data.csv')
print(data.head(10))
print()


#==================== PREPROCESSING =======================================

#checking missing values

print("=====================================================")
print("--------- Before Checking missing values ------------")
print("=====================================================")
print(data.isnull().sum())
print()


print("=====================================================")
print("--------- After Checking missing values ------------")
print("=====================================================")
data=data.fillna(0)
print(data.isnull().sum())
print()

#==== LABEL ENCODING ====

from sklearn import preprocessing

print("-----------------------------------------------------------")
print("================== Before label Encoding ==================")
print("-----------------------------------------------------------")
print()

print(data['area_type'].head(20))


label_encoder = preprocessing.LabelEncoder()


data['area_type']=label_encoder.fit_transform(data['area_type'])
data['availability']=label_encoder.fit_transform(data['availability'])
data['location']=label_encoder.fit_transform(data['location'].astype(str))
data['size']=label_encoder.fit_transform(data['size'].astype(str))
data['society']=label_encoder.fit_transform(data['society'].astype(str))

print("-----------------------------------------------------------")
print("================== After label Encoding ==================")
print("-----------------------------------------------------------")
print()

print(data['area_type'].head(20))


# import numpy as np
data['total_sqft']=data['total_sqft'].replace('-','')

data=data.drop('total_sqft',axis=1)




#========================= DATA SPLITTING ============================

#=== TEST AND TRAIN ===

x=data.drop('price',axis=1)
y=data['price']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print("-----------------------------------------------------------")
print("======================= Data splitting ====================")
print("-----------------------------------------------------------")
print()
print("Total No Of data          :",data.shape[0])
print()
print("Total No of Training data :",X_train.shape[0])
print()
print("Total No of Testing data :",X_test.shape[0])
print()




#========================= CLASSIFICATION ============================

from sklearn.linear_model import Ridge
from sklearn import metrics

#=== ridge regression ===

#initialize the model
ridgeR = Ridge(alpha = 1)

#fitting the model
ridgeR.fit(X_train, y_train)

#predict the model
y_pred = ridgeR.predict(X_test)


print("-----------------------------------------------------------")
print("======================= RIDGE REGRESSION ===================")
print("-----------------------------------------------------------")
print()


mae_ridge=metrics.mean_absolute_error(y_test, y_pred)

print("1.Mean Absolute Error : ",mae_ridge)


#===== Random Forest Regression Model =======

from sklearn.ensemble import RandomForestRegressor
 
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
 
# fit the regressor with x and y data
regressor.fit(X_train, y_train) 

y_pred_rf=regressor.predict(X_test)

print("-----------------------------------------------------------")
print("============== RANDOM FOREST REGRESSION ===================")
print("-----------------------------------------------------------")
print()


mae_rf=metrics.mean_absolute_error(y_pred_rf,y_test)

print("1.Mean Absolute Error :",mae_rf)
print()

#========================= PREDICTION ============================

print("-----------------------------------------------------------")
print("======================= PREDICTION ========================")
print("-----------------------------------------------------------")
print()

for i in range(0,10):
    Results=y_pred_rf[i]
    print("------------------------------------------")
    print()
    print([i],"The predicted house price is ", Results)
    print()


#=============================== PREDICTION ===========================

import numpy as np
print()
print("-------------------------------------------------------------")
print()
print("======== Input data 1 =============")
print()
input_1 = np.array([2,80,671,23,0,7,0]).reshape(1, -1)
print()
print("The Actuall input data is : ",input_1)
predicted_data = ridgeR.predict(input_1)
print()
print("Thepredicted house price is : ", predicted_data)
print()

# graph

import seaborn as sns
sns.barplot(y=[mae_rf,mae_ridge],x=["RF","Ridge"])
