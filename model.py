# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

train = pd.read_csv('C:/Users/YallaKOREA/Desktop/CODE/project/section3/data/train.csv')
del train['일자']
del train['조식메뉴']
del train['중식메뉴']
del train['석식메뉴']

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

x_train = train[['요일', '본사정원수', '본사출장자수']]
y_train = train['중식계']

data = {'요일': [0],'본사정원수': [0],'본사출장자수': [0]}
x_test = pd.DataFrame(data)   

model = RandomForestRegressor()
model.fit(x_train, y_train)

joblib.dump(model, './model/model.pkl')

# predict = model.predict(x_test)
# print(predict)

# # Saving model to disk
# pickle.dump(regressor, open('model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2, 9, 6]]))