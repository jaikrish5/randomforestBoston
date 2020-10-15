#Imports 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
import pickle

# Import boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)



# creating a boston dataframe
df= pd.DataFrame(boston.data,columns=boston.feature_names)

#setting features for model
x = df[['RM','PTRATIO','LSTAT']]

# # appending price to it
# df['Price'] = boston.target

# corrmat = df.corr()
# corrmat
# target variable
y= pd.DataFrame(data=boston.target)


y=y.astype('int')
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

 # create regressor object 
model = RandomForestRegressor(n_estimators = 10, random_state = 0) 
  
# fit the regressor with x and y data 
model.fit(x, y)  




pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


#y_predict = model.predict(X_test)