#!/usr/bin/env python
# coding: utf-8

import os
import random
import sys

import xml
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import math 
from sklearn.metrics import confusion_matrix

class Printer():
    def __init__(self, obj, total):
        self.idx = 0 
        self.obj = obj
        self.total = total
        
    def progressbar(self, total, i, bar_length=50, prefix=""):
        dot_n = int(i/total*bar_length)
        dot = '>'*dot_n
        empty = '_'*(bar_length-dot_n)
        sys.stdout.write(f"\r {prefix} [{dot}{empty}] {i/total*100:3.2f} % Done")
        if i == total:
            sys.stdout.write("\n")
        
    def __next__(self):
        self.progressbar(self.total, self.idx, bar_length=50, prefix="")
        self.idx += 1
        return next(self.obj)


class pbar():
    def __init__(self, literable):
        self.literable = literable
        self.total = len(self.literable)
    
    def __iter__(self):
        liter = iter(self.literable)
        return Printer(liter,self.total)


def from_xml(file, idx=-1):
    base_name = os.path.basename(file)
    base_name = base_name.split('.')[0]
    
    with open(file) as f:
        string = f.read()
    
    tree = ET.ElementTree(ET.fromstring(string))
    note = tree.getroot()

    drug_code =[]
    dosage = []
    amount = []
    
    p = note.findall('KIMSPOCParam')[idx]
    for rx in p.find('Diagnosis').find("RxInfo").findall("Rx"):
        cnt = rx.attrib["Count"]
        if cnt == '':
            cnt = 1
        else:
            cnt = np.int(cnt)
        
        amt = np.float(rx.attrib["Amount"])
        if amt == 0.0:
            amt = 1
        
        drug_code.append(rx.attrib['Code'].replace(' ',''))
        dosage.append(rx.attrib['DosageCode'].replace(' ',''))
        amount.append(amt)
    return drug_code, dosage, amount


# def progressbar(total, i, bar_length=50, prefix=""):
#     dot_n = int(i/total*bar_length)
#     dot = '>'*dot_n
#     empty = '_'*(bar_length-dot_n)
#     sys.stdout.write(f"\r {prefix} [{dot}{empty}] {i/total*100:3.2f} % Done")
#     if i == total:
#         sys.stdout.write("\n")
        
def index_replacement(codes, code_mapper=None):
    # get unique value from multi list
    code_list = []
    for code in codes:
        for c in code:
            code_list.append(c)
    code_list = set(code_list)
    
    # create mapper
    if code_mapper is None:
        code_mapper = {}
        for i, d in enumerate(code_list):
            code_mapper[d] = i + 1
        
    for i, code in enumerate(codes):
        for j , c in enumerate(code):
            try:
                codes[i][j] = code_mapper[c]
            except KeyError:
                print(f"except code {c}")
                codes[i].remove(c)
    return codes, code_mapper


## define function
def date_parser(string, format_string='%Y%m%d%H%M%S'):
    return pd.datetime.strptime(string, format_string)


def train_test_random_split(df, rate = 0.3):
    df_index = list(df.index)
    random.shuffle(df_index)

    size = int(len(df_index)*rate)
    test_idx = df_index[:size]
    train_idx = df_index[size:]

    return df.loc[train_idx], df.loc[test_idx]


def get_age(a,b):
    year = a.year - b.year
    month = a.month - b.month
    return year*12 + month


def angle(p):
    x = p[0]
    y = p[1]
    return 180*math.atan2(x,y)/math.pi


def last_checkpoint(checkpoint_dir):
    checkpoints = [i for i in os.listdir(checkpoint_dir) if "checkpoint" in i]
    checkpoints.sort(key = lambda s : os.path.getmtime(os.path.join(checkpoint_dir,s)))
    return os.path.join(checkpoint_dir, checkpoints[-1])
    

def print_confusion_matrix(ytest, ypred, cut_off):
    """confusion_matrix 출력 format
    
    Arguments:
        ytest {numpy array} -- label 1 or 0
        ypred {numpy array} -- predicted score 0~1
        cut_off {int} -- cut off score
    """
    tp,fn,fp,tn = confusion_matrix(ytest,
                     ypred>cut_off,
                    labels=[1,0]).ravel()
    print(format('cut off :','15s'), format(cut_off,'<15.2f'))
    print("""    {}{}{}
    {}{}{}
    {}{}{}
    """.format(format('Predict \\ True','15s'),format(1,'10.0f'),format(0,'10.0f'),
              format('1','>15s'), format(tp,'10.0f'), format(fp,'10.0f'),
               format('0','>15s'), format(fn,'10.0f'), format(tn,'10.0f')
              ))
    recall = tp/(tp+fn)*100 if (tp+fn) != 0 else 0
    precision = tp/(tp+fp)*100 if (tp+fp) != 0 else 0
    print(format('recall :','15s'),format(recall, '<15.3f'))
    print(format('precision :','15s'),format(precision, '<15.3f'))
    print(format('f1_score :','15s'),format(2*tp/(2*tp+fp+fn)*100, '<15.3f'))
    
    
def labeling(frame):
    # frame.loc[frame.label == 1, 'label'] = 0
    frame = frame.assign(label = 0)
    
    frame.loc[(frame.Drugcode == 'PLDRO-S') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PLDRO-S') & (frame.Amount / frame.PatientWeight > 0.7),'label'] = 1
    frame.loc[(frame.Drugcode == 'PMTK42-P') & (frame.age < 72) & (frame.Count >= 2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PMTK52') & (frame.age >= 72) & (frame.Count >= 2), 'label'] = 1
    frame.loc[(frame.Drugcode == 'PDXB-S') & (frame.Amount < 1), 'label'] = 1
    frame.loc[(frame.Drugcode == 'PDXB-S') & (frame.Amount / frame.PatientWeight < 0.2), 'label'] = 1
#     frame.loc[(frame.Drugcode == 'PDXB-S') & (frame.Amount / frame.PatientWeight > 0.7), 'label'] = 1
    frame.loc[(frame.Drugcode == 'PCPDX-S') & (frame.Amount / frame.PatientWeight < 0.1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PCPDX-S') & (frame.Amount / frame.PatientWeight > 0.7),'label'] = 1
    frame.loc[(frame.Drugcode == 'PIBP-S') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PIBP-S') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
#     frame.loc[(frame.Drugcode == 'PDS-S1') & (frame.Amount / frame.PatientWeight < 0.5),'label'] = 1   ## → label 0.0만 27개. 1.0 없음
    frame.loc[(frame.Drugcode == 'PIVY-S') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PIVY-S') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PIVY-S') & (frame.Amount / frame.PatientWeight > 0.7),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAMCLN-S') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAMCLN-S') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAMCLN-S') & (frame.Amount / frame.PatientWeight > 0.7),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAMCL-S2') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAMCL-S2') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAMCL-S2') & (frame.Amount / frame.PatientWeight > 0.7),'label'] = 1
    
    dos = ['530000','530100','530300','530500','531000','530104','530204','530304','530310','530313','530404','530504','530513','530910','531004','531013','531110','530110','530113','530200','530210','530213','530400','530410','530510','530900','530904','530913','531010','531100','531104','531113','530413']
    drug = ['PFEN20-N', 'PMMTY6-N', 'PCL-N', 'PFEN10-N', 'PMMTP-N', 'PHOFTC-N', 'PFEN5-N', 'PHO-PDSM']
    frame.loc[(frame.Drugcode.isin(drug) == True) & (frame.Dosage.isin(dos) == False), 'label'] = 1
    frame.loc[(frame.Drugcode.isin(drug) == True) & (frame.Count >= 3),'label'] = 1
    
    frame.loc[(frame.Drugcode == 'PAZM1-S') & (frame.Count >= 2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAZM1-S') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAZM1-S') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PAZM1-S') & (frame.Amount / frame.PatientWeight > 1),'label'] = 1
    
    frame.loc[(frame.Drugcode == 'POMP1') & (frame.Count >= 2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PPLG-S') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PPLG-S') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PPLG-S') & (frame.Amount / frame.PatientWeight > 0.7),'label'] = 1
    frame.loc[(frame.Drugcode == 'PTMB-S') & (frame.Amount < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PTMB-S') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PTMB-S') & (frame.Amount / frame.PatientWeight > 0.7),'label'] = 1
    frame.loc[(frame.Drugcode == 'PCFX-P') & (frame.Amount*50 / frame.PatientWeight < 1),'label'] = 1
    frame.loc[(frame.Drugcode == 'PCFX-P') & (frame.Amount*50 / frame.PatientWeight > 4),'label'] = 1
    frame.loc[(frame.Drugcode == 'PDPR-S') & (frame.Amount / frame.PatientWeight < 0.2),'label'] = 1
    frame.loc[(frame.Drugcode == 'PDPR-S') & (frame.Amount / frame.PatientWeight > 0.5),'label'] = 1
    
    return frame
