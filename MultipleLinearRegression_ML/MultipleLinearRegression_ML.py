#loading libraries
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

# loading data
boston = datasets.load_boston(return_X_y=False)
X=boston.data
y=boston.target

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)

# create model
MLReg = linear_model.LinearRegression()

# train the model
MLReg.fit(X_train, y_train)

# predict
y_pred = MLReg.predict(X_test)
print(y_pred[:3])
print(y_test[:3])

# get report 
my_coef = MLReg.coef_
my_varriance = MLReg.score(X_test,y_test)
print("coefficient from our model is: ", my_coef)
print("varrience score from our model is: ", my_varriance)