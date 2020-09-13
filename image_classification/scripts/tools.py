# -*- coding: utf-8 -*-

import numpy as np
import os, glob, skimage
import pickle

def set_GPU(device_num):
    """
    Description :
        GPU 사용시, 특정 GPU만을 사용할 수 있도록 설정
        
    Arguments :
        - device_num : 사용할 GPU 넘버, (dtpye : str)
        
    """
    # 데이터 타입이 str이 아닌경우 변환
    if type(device_num) is not str:
        device_num = str(device_num)
        
    # GPU 환경설정
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=device_num

def write_pickle(path, data):
    """
    Description :
        Pickle 파일 포멧 형태로 주어진 데이터 저장
        
    Arguments :
        - path : 데이터를 저장할 경로와 파일 이름
        - data : pickle 형태로 저장할 데이터
        
    """
    with open(path, 'wb') as fout:
        pickle.dump(data, fout)
        
    print('your data have just written at {}.'.format(path))
    
def read_pickle(path):
    """
    Description :
        Pickle 파일을 읽어와 데이터 출력
        
    Arguments :
        - path : 읽어올 pickle 데이터 경로
        
    Output : 
        pickle안에 들어 있는 데이터
    """
    with open(path, 'rb') as fin:
        data = pickle.load(fin)
    return data