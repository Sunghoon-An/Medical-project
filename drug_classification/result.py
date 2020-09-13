#!/usr/bin/env python
# coding: utf-8

import sys
sys.path = ["/product/src/gruads/anaconda3/envs/clone/lib/python3.7/site-packages"]+sys.path
#import matplotlib.pyplot as plt
import os
import time
import pickle
import math 
import datetime
import argparse
import warnings
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import xgboost
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
import gensim
from gensim.models import Word2Vec

from config import *
from utils.util import *
# from utils.callback_util import *

def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-f','--fold',type= str
                        , help = 'fold', required=True) 
    parser.add_argument('-w','--weight',type= str, help = 'checkpoint to restore') 
    arg = parser.parse_args() 
    return arg

dt = datetime.datetime.now()
dt = dt.strftime("%Y-%m-%d %H:%M:%S")

# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# ERROR
tf.get_logger().setLevel(logging.CRITICAL)
# tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["KMP_WARNINGS"] = 'off'
warnings.filterwarnings("ignore")

fold = args().fold
weight = args().weight

x_test = np.load(os.path.join(BACKUP, fold,'x_test.npy'))
y_test = np.load(os.path.join(BACKUP, fold,'y_test.npy'))

x_train = np.load(os.path.join(BACKUP, fold,'x_train.npy'))
y_train = np.load(os.path.join(BACKUP, fold,'y_train.npy'))

with open(os.path.join(RESULT,f'model_{fold}.json'), 'r') as f:
    model = tf.keras.models.model_from_json(f.read())
    
model.load_weights(os.path.join(RESULT, fold, weight))

print("==============================================")
print(f"GruPEP Ver1.0 {dt}")
print(f"test : {fold}")
print("==============================================")

y_pred = model.predict(x_test, verbose = 1, batch_size =256)
error_df = pd.DataFrame({'y_pred':y_pred.reshape(-1)
                         , "Class":y_test.reshape(-1)} ) 

# error_df = pd.DataFrame({'y_pred':y_pred, "Class":y_test} ) #
error_df = error_df.sample(error_df.shape[0])
error_df=error_df.reset_index(drop=True)

print_confusion_matrix(error_df.Class,error_df.y_pred, cut_off=0.1)