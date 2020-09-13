#!/usr/bin/env python
# coding: utf-8
```보완상 $$$ Masking```


import sys
sys.path = ["/$$$/$$$"]+sys.path
import math
import warnings

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
import xgboost
# import matplotlib.pyplot as plt

from config import *
from utils.util import *

warnings.filterwarnings("ignore")

baseinfo = pd.read_csv(os.path.join(DATA,"$$$.csv"), index_col =0)

baseinfo = baseinfo.assign(date = baseinfo.Order.apply(lambda x : date_parser(x[:8], format_string='%Y%m%d')) )
baseinfo.Birth = baseinfo.Birth.apply(lambda x : date_parser(x[:7],"%Y-%m"))
baseinfo = baseinfo.assign(age = [get_age(b,d) for b, d in zip(baseinfo.date, baseinfo.PatientBirth)])

baseinfo[baseinfo.subject=='$$$'].$$$Num.nunique()

tmp = baseinfo[baseinfo.subject=='$$$'].groupby('$$$Num')[["Height","Weight","Sex","age"]].max()

#  preprocess 
tmp.loc[tmp.Weight == 0,"Weight"] = None
tmp.loc[tmp.Height == 0,"Height"] = None
tmp.loc[tmp.Height>200,"Height"] =None
tmp.loc[(tmp.Weight > 400), "Weight"] = tmp.loc[tmp.Weight > 400, "Weight"]/1000
tmp.loc[tmp.Height<2.5,"Height"] = None


tmp = tmp.assign(angle=[angle((a,b)) for a,b in zip(tmp.Height, tmp.Weight)])
tmp.loc[tmp.angle.apply(lambda x : True if x < 55 else False),['Height','Weight']]= None

tmp = tmp.assign(angle=[angle((a,b+65)) for a,b in zip(tmp.Height, tmp.Weight)])
tmp.loc[tmp.angle.apply(lambda x : True if x > 65 else False),['Height','Weight']]= None

tmp = tmp.assign(angle=[angle((a,b+15)) for a,b in zip(tmp.Height, tmp.Weight)])
tmp.loc[tmp.angle.apply(lambda x : True if x < 55 else False),['Height','Weight']]= None

dataset = tmp[tmp.Weight.notnull() &tmp.Height.notnull()]
dataset.Sex = dataset.Sex.apply(lambda x : 1 if x=='M' else 0)


# ## XGBRegressor
train, test = train_test_split(dataset,test_size =0.1)
scaler = StandardScaler()

y_train = train.Height
x_train = train[['Sex', 'age']]

y_test = test.Height
x_test = test[['Sex', 'age']]

## scale x
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## scale y
patient_info = {"Height" : {"mean" : train.Height.mean()
                            ,"std" : train.Height.std()}
                  }
y_train = y_train.apply(lambda x : (x - $$$_info["Height"]['mean'])/$$$_info["Height"]['std'])
y_test = y_test.apply(lambda x : (x - $$$_info["Height"]['mean'])/$$$_info["Height"]['std'])

## model 
xgb = xgboost.XGBRegressor(max_depth=10,learning_rate=1,n_estimators=100)
xgb.fit(x_train, y_train)

print(f"train loss {np.mean((xgb.predict(x_train)-y_train)**2)}")
print(f"test loss {np.mean((xgb.predict(x_test)-y_test)**2)}")

## save
xgb.save_model(os.path.join(RESULT,"XGBRegressor_Height.model"))

with open(os.path.join(RESULT,"scaler_Height.pkl"),'wb') as f:
    pickle.dump(scaler, f)

## XGBRFRegressor
train, test = train_test_split(dataset,test_size =0.1)

y_train = train.Weight
x_train = train[['Height','Sex', 'age']]
y_test = test.Weight
x_test = test[['Height','Sex', 'age']]

## scale x
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## scale y
patient_info["weight"] = {"mean" : train.Weight.mean()
                         ,"std" : train.Weight.std()}
y_train = y_train.apply(lambda x : (x - $$$_info["weight"]['mean'])/$$$_info["weight"]['std'])
y_test = y_test.apply(lambda x : (x - $$$_info["weight"]['mean'])/$$$_info["weight"]['std'])

## model
xgb = xgboost.XGBRFRegressor(max_depth=10,learning_rate=1,n_estimators=300)
xgb.fit(x_train, y_train)

print(f"train loss {np.mean((xgb.predict(x_train)-y_train)**2)}")
print(f"test loss {np.mean((xgb.predict(x_test)-y_test)**2)}")

## save
xgb.save_model(os.path.join(RESULT,"XGBRFRegressor_Weight.model"))

with open(os.path.join(RESULT,"scaler_Weight.pkl"),'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(RESULT, "patient_info.dict"), 'w') as f:
    f.write(str($$$_info))
