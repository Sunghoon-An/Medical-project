#!/usr/bin/env python
# coding: utf-8

import sys
sys.path = ["/product/src/gruads/anaconda3/envs/clone/lib/python3.7/site-packages"]+sys.path
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

# rxinfo = pd.read_csv(os.path.join(DATA,"rx_info.csv"), index_col =0)
baseinfo = pd.read_csv(os.path.join(DATA,"prescript_info.csv"), index_col =0)

baseinfo = baseinfo.assign(date = baseinfo.OrderNum.apply(lambda x : date_parser(x[:8], format_string='%Y%m%d')) )
baseinfo.PatientBirth = baseinfo.PatientBirth.apply(lambda x : date_parser(x[:7],"%Y-%m"))
baseinfo = baseinfo.assign(age = [get_age(b,d) for b, d in zip(baseinfo.date, baseinfo.PatientBirth)])

baseinfo[baseinfo.subject=='A41'].PatientNum.nunique()

tmp = baseinfo[baseinfo.subject=='A41'].groupby('PatientNum')[["PatientHeight","PatientWeight","PatientSex","age"]].max()

#  preprocess 
tmp.loc[tmp.PatientWeight == 0,"PatientWeight"] = None
tmp.loc[tmp.PatientHeight == 0,"PatientHeight"] = None
tmp.loc[tmp.PatientHeight>200,"PatientHeight"] =None
tmp.loc[(tmp.PatientWeight > 400), "PatientWeight"] = tmp.loc[tmp.PatientWeight > 400, "PatientWeight"]/1000
tmp.loc[tmp.PatientHeight<2.5,"PatientHeight"] = None


tmp = tmp.assign(angle=[angle((a,b)) for a,b in zip(tmp.PatientHeight, tmp.PatientWeight)])
tmp.loc[tmp.angle.apply(lambda x : True if x < 55 else False),['PatientHeight','PatientWeight']]= None

tmp = tmp.assign(angle=[angle((a,b+65)) for a,b in zip(tmp.PatientHeight, tmp.PatientWeight)])
tmp.loc[tmp.angle.apply(lambda x : True if x > 65 else False),['PatientHeight','PatientWeight']]= None

tmp = tmp.assign(angle=[angle((a,b+15)) for a,b in zip(tmp.PatientHeight, tmp.PatientWeight)])
tmp.loc[tmp.angle.apply(lambda x : True if x < 55 else False),['PatientHeight','PatientWeight']]= None

dataset = tmp[tmp.PatientWeight.notnull() &tmp.PatientHeight.notnull()]
dataset.PatientSex = dataset.PatientSex.apply(lambda x : 1 if x=='M' else 0)


# ## XGBRegressor
train, test = train_test_split(dataset,test_size =0.1)
scaler = StandardScaler()

y_train = train.PatientHeight
x_train = train[['PatientSex', 'age']]

y_test = test.PatientHeight
x_test = test[['PatientSex', 'age']]

## scale x
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## scale y
patient_info = {"Height" : {"mean" : train.PatientHeight.mean()
                            ,"std" : train.PatientHeight.std()}
                  }
y_train = y_train.apply(lambda x : (x - patient_info["Height"]['mean'])/patient_info["Height"]['std'])
y_test = y_test.apply(lambda x : (x - patient_info["Height"]['mean'])/patient_info["Height"]['std'])

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

y_train = train.PatientWeight
x_train = train[['PatientHeight','PatientSex', 'age']]
y_test = test.PatientWeight
x_test = test[['PatientHeight','PatientSex', 'age']]

## scale x
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## scale y
patient_info["weight"] = {"mean" : train.PatientWeight.mean()
                         ,"std" : train.PatientWeight.std()}
y_train = y_train.apply(lambda x : (x - patient_info["weight"]['mean'])/patient_info["weight"]['std'])
y_test = y_test.apply(lambda x : (x - patient_info["weight"]['mean'])/patient_info["weight"]['std'])

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
    f.write(str(patient_info))