#!/usr/bin/env python
# coding: utf-8

import sys
sys.path = ["/home/gruads/anaconda3/envs/clone/lib/python3.7/site-packages"]+sys.path
import os
import logging

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, TimeDistributed, Reshape, Permute, Flatten
from tensorflow.keras.layers import Lambda, concatenate, multiply
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import glorot_uniform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# from config import *

def self_attention(x):
    n = int(x.get_shape()[1])
    a = Dense(n, activation='sigmoid' #, name = f'attention_{i}'
              ,kernel_initializer='he_uniform')(x)
    return multiply([x, a])


def fc(x, n, batch = False, attention = False, acti = 'relu'):
    x = Dense(n, activation = acti, kernel_initializer= 'he_uniform')(x)
    x = BatchNormalization()(x) if batch else x
    x = self_attention(x) if attention else x
    return x


def dnn_model(input_dim, fs):
    inputs = Input(shape=(input_dim,))
    
    x1 = Lambda(lambda x : x[:,:(input_dim - fs)], name = "feature")(inputs)
    x2 = Lambda(lambda x : x[:,-fs:], name = "Drug_info")(inputs)
    x2 = fc(x2, int(fs/2), batch = False, attention = False, acti = 'relu')
    x = concatenate([x1,x2], axis = -1)
#     input2 = Input(shape=(input_dim2,))
    node= 512
    
    x = Dense(input_dim, activation='selu'
              ,kernel_initializer='he_uniform'
              ,use_bias=True
              ,kernel_constraint=max_norm(3)
              )(x)
    x = self_attention(x)
    
    for _ in range(4):
        x = fc(x, node, batch = True, attention = True, acti = 'selu')
    
    out1 = Dense(1, activation='sigmoid'
                 ,kernel_initializer='glorot_uniform')(x)
    
    return Model(inputs, out1, name='classifier')


def chunje_1(input_dim):
    model = Sequential()
    model.add(Reshape((1,-1), input_shape = (input_dim,)))
    model.add(LSTM(16, kernel_initializer = 'he_uniform', activation = 'selu', recurrent_regularizer = keras.regularizers.l1(0.01),
                  bias_regularizer = None, activity_regularizer = None, dropout = 0.3, 
                  recurrent_dropout = 0.3, return_sequences = True))
    for i in range(3):
        model.add(LSTM(8, activation = 'selu', return_sequences = True))
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    return model

    
if __name__ == "__main__":
    # FEATURE_SIZE = 10
    # model = creat_autoencoder(1200)
    model = dnn_model(27,10)
    model.summary()
    
