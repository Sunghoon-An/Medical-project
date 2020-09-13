#!/usr/bin/env python
# coding: utf-8

import sys
sys.path = ["/$$$/$$$"]+sys.path
import os
import time
import pickle
import argparse
import warnings

#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from imblearn.under_sampling import TomekLinks, ClusterCentroids, RandomUnderSampler
from imblearn.over_sampling import ADASYN, SVMSMOTE
from imblearn.pipeline import make_pipeline

import tensorflow 
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam, Adagrad, Adadelta
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint, TensorBoard

from config import *
from utils.util import *
from models.dnn import *

def args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-f','--fold', type= str
                        , help = 'fold', required=True) 
    parser.add_argument('-r','--restore',action= 'store_true', help = 'restore checkpoint') 
    parser.add_argument('-e','--epoch', type= int, required=True, help = 'epoch') 
    arg = parser.parse_args() 
    return arg

args = args()
fold = args.fold

## 텐서보드용 log 디렉토리 생성
path = os.path.join("graph", fold)
if os.path.isdir(path) == False:
    os.mkdir(path)

## create result dir
path = os.path.join("result", fold)
if os.path.isdir(path) == False:
    os.mkdir(path)

warnings.filterwarnings("ignore")

print('load train/test data')
x_test = np.load(os.path.join(BACKUP, fold, "x_test.npy"))
y_test = np.load(os.path.join(BACKUP, fold, "y_test.npy"))

x_train = np.load(os.path.join(BACKUP, fold, "x_train.npy"))
y_train = np.load(os.path.join(BACKUP, fold, "y_train.npy"))

## ------------ train ----------------
# K.clear_session()
batchsize = 128
def auc(y_test, y_pred):
    auc = tf.metrics.auc(y_test, y_pred,
        curve = 'PR',
        summation_method = 'careful_interpolation',
        num_thresholds = 500
        )[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def recall(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    

pipe = make_pipeline(
    RandomUnderSampler(sampling_strategy=0.2
              , random_state=42)
    ,SVMSMOTE(sampling_strategy=0.7
             ,n_jobs = 12, random_state=42)
)

start = datetime.datetime.now()
print(f"{start} : over sampling......")
print(f"1 : {np.sum(y_train)}, 0 :{len(y_train)- np.sum(y_train)}")
x_tmp , y_tmp = pipe.fit_resample(x_train, y_train)
print(f"sampling : {datetime.datetime.now() - start}")
print(f"1 : {np.sum(y_tmp)}, 0 :{len(y_tmp)- np.sum(y_tmp)}")


sgd = SGD(lr = 0.0002, momentum = 0.9, decay = 10e-6, nesterov = True)
adam = Adam(lr = 1e-4, decay = 1e-6)
rmsprop = RMSprop(lr = 1e-5, epsilon = 1.0)
amsgrad = Adam(lr = 1e-5, decay = 1e-6, amsgrad = True)
adagrad = Adagrad(lr = 1e-5, decay = 1e-6)
nadam = Nadam(lr = 0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
adadelta = Adadelta(lr = 1., rho=0.95, epsilon=None, decay=0.0)

model = dnn_model(x_train.shape[1], FEATURE_SIZE)
# model = chunje_1(x_train.shape[1])
model.compile(optimizer = adadelta, loss = 'binary_crossentropy'
              , metrics = [tf.keras.metrics.AUC()]) #  recall, precision
fold
callbacks = [
# EarlyStopping(monitor = 'val_auc', patience = 20, verbose = 1, mode = 'max'),
    ModelCheckpoint(os.path.join(RESULT, fold, 'model_checkpoint-{epoch:03d}-{val_auc:0.3f}.h5')
    , monitor = 'val_auc', save_best_only = True, mode = 'max'),
    TensorBoard(log_dir = os.path.join("graph",fold), histogram_freq=0, write_graph=True, write_images=True) 
    ]

model_json = model.to_json()
with open(os.path.join(RESULT, f'model_{fold}.json'), 'w')as json_file:
    json_file.write(model_json)
    
model.summary()
if args.restore:
    chkp = last_checkpoint(os.path.join(RESULT, fold))
    model.load_weights(chkp)
    init_epoch = int(os.path.basename(chkp).split('-')[1])
    print("================== restore checkpoint ==================")
else :
    init_epoch = 0
    print("================== restart train ==================")

print("Training......")
hist = model.fit(x = x_train, y = y_train
    , epochs = args.epoch
    , verbose = 1
    , validation_data = (x_test, y_test)
    , shuffle = True
    , callbacks = callbacks
    , initial_epoch = init_epoch
    )

