#!/usr/bin/env python
# coding: utf-8
``` 모든 데이터와 Root는 보완상 $로 표시```

import pickle
import pandas as pd
import numpy as np
import re
import warnings

from config import *
from utils.util import *

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", -1)
pd.set_option("display.max_columns", None)

def date_parser(string, format_string):
    # %Y%m%d%H%M%S
    return pd.datetime.strptime(string, format_string)

""" preprocess prescript_info """
df = pd.read_csv(os.path.join(DATA,'prescript_info.csv'), index_col =0)
## OrderNum 앞 8자리 날짜 + InTime 앞 8자리 시간
df = df.assign(indatetime = [o[:8] +" "+ i[:8] for o,i in zip(df.$$$, df.$$$)])
## 날짜 형식으로 변환
df.indatetime = df.indatetime.apply(lambda x : date_parser(x, "%Y%m%d %H:%M:%S"))
df = df.assign(Date = df.OrderNum.apply(lambda x : date_parser(str(x)[:8], "%Y%m%d")))
print(df.shape)

rxinfo = pd.read_csv(os.path.join(DATA,'$$$.csv'), index_col =0)

## $$$에서 $$$만 추출
rxinfo = rxinfo.assign(person_id = [l[11:17] for l in rxinfo.Basename])

""" preprocess $$$ """
## $$$ 날짜만 추출
rxinfo = rxinfo.assign(date = [l[:8] for l in rxinfo.Basename])
## $$$ $$$만 추출
rxinfo = rxinfo.assign(subject = [l[8:11] for l in rxinfo.Basename])
## 날짜 데이터 숫자형으로 변환 
rxinfo.Days = pd.to_numeric(rxinfo.Days, errors='coerce')
rxinfo.$$$ = rxinfo.$$$.apply(lambda x : x.replace(" ", ""))
print(rxinfo.shape)

## $$$정보( ex) $$$ )
ms = pd.read_csv(os.path.join(DATA,'$$$.csv'), encoding='iso-8859-1')

""" preprocess measurement """
ms.rename(columns={'PERSON_ID' : 'PatientNum'}, inplace=True)
## 일자 date 형식으로 변환
ms.MEASUREMENT_DATE = ms.MEASUREMENT_DATE.apply(lambda x : date_parser(x, '%Y/%m/%d'))
ms.rename(columns={'MEASUREMENT_DATE':'Date'}, inplace=True)
## 같은 일자에 검사를 여러번 한경우가 있어서 마지막 것만 사용
ms = ms.drop_duplicates(["$$$","$$$","$$$"], keep ='last')
print(ms.shape)


## $$$정보를 $$$테이블에 조인
for code in [$$$,$$$,]:
    tmp = ms[ms.$$$==code][["$$$","Date","VALUE_AS_NUMBER"]]
    tmp = tmp.rename(columns={"VALUE_AS_NUMBER":f"{code}"})
    df = df.merge(tmp, how='left', on=['PatientNum', 'Date'])

## $$$결과가 FAILED 인경우가 있어서 해당경우는 Null 처리
df.loc[df["12010101"] == "FAILED" , "12010101"] = None


""" fill null value """
## $$$가 없다는 것은 $$$를 하지 않은경우로 정상치라 가정하고 해당 $$$의 정상범위 안에서 랜덤값을 채워 넣음
for code in [210112,210109,210114,210113,210115,210111,12010101,210105]:
    if code == 210112:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([x for x in range(5, 41)], size=len(df.index))))
        
    elif code == 210109:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([round(x, 1) for x in list(np.arange(3.6, 5.5, 0.1))], size=len(df.index))))
        
    elif code == 210114:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([round(x, 1) for x in list(np.arange(0.0, 0.4, 0.1))], size=len(df.index))))
        
    elif code == 210113:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([round(x, 1) for x in list(np.arange(0.2, 1.1, 0.1))], size=len(df.index))))
        
    elif code == 210115:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([round(x, 1) for x in list(np.arange(61))], size=len(df.index))))
        
    elif code == 210111:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([round(x, 1) for x in list(np.arange(5, 36, 1))], size=len(df.index))))
        
    elif code == 12010101:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([round(x, 1) for x in list(np.arange(10.0, 15.1, 0.1))], size=len(df.index))))
        
    elif code == 210105:
        df[f'{code}'] = df[f'{code}'].fillna(pd.Series(np.random.choice([round(x, 1) for x in list(np.arange(0.3, 1.1, 0.1))], size=len(df.index))))
        
## $$$결과 값에 문자열이 많아 제거
func1 = lambda x : str(x).replace("> ","").replace("< ","")

vec_flo = ['210112','210115','210111','210109','210114','210113','12010101','210105']
for col in vec_flo:
    df[col] = df[col].apply(func1)
    print(col)
    df[col] = df[col].astype(np.float32)


df.InTime = df.InTime.apply(lambda x : 
                            pd.Timedelta(str(date_parser(str(x)[:8],"%H:%M:%S"))[-8:]) )

df = df.sort_values(["Basename","Idx"])
tmp = df.groupby(["Basename"]).InTime.diff(periods=1)
tmp = tmp.to_list() # 1599100
df = df.assign(timediff= tmp[1:] + [tmp[0]] )

## 학습에 사용될 컬럼들만 가져옴 status 컬럼은 이름이 중복됨으로 pres_status로 변경
df = df[['$$$','$$$','$$$']].rename(columns={"status":"pres_status"})

## $$$ + $$$ Merge
rxinfo = pd.merge(rxinfo, df   
        ,how='left'
        ,on =['Basename','Idx']
        ,validate = "many_to_one")

## 각 $$$정보의 최종$$$만 정상데이터로 가정
tmp = rxinfo.groupby("Basename").Idx.max().reset_index()

error = ["$$$"
        ,"$$$"
        ,"$$$"
        ,"$$$"]
tmp = tmp[tmp.Basename.isin(error) ==False]

## 각 $$$정보에 최종$$$ 정상데이터
tmp = tmp.assign(normal = 0)

rxinfo = pd.merge(rxinfo, tmp
        ,how='left'
        ,on =['Basename','Idx']
        ,validate = "many_to_one")

## 저장
rxinfo.to_csv(os.path.join(BACKUP,'frame.csv'), index=False)


