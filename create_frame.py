#!/usr/bin/env python
# coding: utf-8

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
df = df.assign(indatetime = [o[:8] +" "+ i[:8] for o,i in zip(df.OrderNum, df.InTime)])
## 날짜 형식으로 변환
df.indatetime = df.indatetime.apply(lambda x : date_parser(x, "%Y%m%d %H:%M:%S"))
df = df.assign(Date = df.OrderNum.apply(lambda x : date_parser(str(x)[:8], "%Y%m%d")))
print(df.shape)

rxinfo = pd.read_csv(os.path.join(DATA,'rx_info.csv'), index_col =0)
## (Basename - RSA(혹은 TRSA))= 처방번호 (* RSA는 암호화방식으로 원래는 없던것 T는 퇴원처방)
## 처방번호 = 날짜(20181201) + 진료과(A41) + 환자번호(123456)
## 처방번호에서 환자번호만 추출
rxinfo = rxinfo.assign(person_id = [l[11:17] for l in rxinfo.Basename])

""" preprocess rxinfo """
## 처방번호 날짜만 추출
rxinfo = rxinfo.assign(date = [l[:8] for l in rxinfo.Basename])
## 처방번호 진료과만 추출
rxinfo = rxinfo.assign(subject = [l[8:11] for l in rxinfo.Basename])
## 날짜 데이터 숫자형으로 변환 
rxinfo.Days = pd.to_numeric(rxinfo.Days, errors='coerce')
rxinfo.Drugcode = rxinfo.Drugcode.apply(lambda x : x.replace(" ", ""))
print(rxinfo.shape)

## 진단정보( ex) 사구체여과율, 혈압, 혈당 ... )
ms = pd.read_csv(os.path.join(DATA,'measurement.csv'), encoding='iso-8859-1')

""" preprocess measurement """
ms.rename(columns={'PERSON_ID' : 'PatientNum'}, inplace=True)
## 검사일자 date 형식으로 변환
ms.MEASUREMENT_DATE = ms.MEASUREMENT_DATE.apply(lambda x : date_parser(x, '%Y/%m/%d'))
ms.rename(columns={'MEASUREMENT_DATE':'Date'}, inplace=True)
## 같은 검사일에 검사를 여러번 한경우가 있어서 마지막 것만 사용
ms = ms.drop_duplicates(["PatientNum","Date","MEASUREMENT_SOURCE_VALUE"], keep ='last')
print(ms.shape)


## 검사정보를 진단테이블에 조인
for code in [210112,210109,210114,210113,210115,210111,12010101,210105]:
    tmp = ms[ms.MEASUREMENT_SOURCE_VALUE==code][["PatientNum","Date","VALUE_AS_NUMBER"]]
    tmp = tmp.rename(columns={"VALUE_AS_NUMBER":f"{code}"})
    df = df.merge(tmp, how='left', on=['PatientNum', 'Date'])

## 검사결과가 FAILED 인경우가 있어서 해당경우는 Null 처리
df.loc[df["12010101"] == "FAILED" , "12010101"] = None


""" fill null value """
## 검사결과가 없다는 것은 검사를 하지 않은경우로 정상치라 가정하고 해당 검사의 정상범위 안에서 랜덤값을 채워 넣음
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
        
## 감사결과 값에 문자열이 많아 제거
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
df = df[['Basename','Idx','PatientBirth','PatientHeight','PatientWeight','PatientSex','status','PrscDrLcsNo','disease_code','InTime','OutTime','timediff','210112','210109', '210114', '210113', '210115', '210111', '12010101', '210105']].rename(columns={"status":"pres_status"})

## 컬럼           내용
# -------------------------------------------------------------------------
# Basename       파일명(처방번호)
# Idx            처방번호내 로그 번호
# PatientBirth   환사 생년
# PatientHeight  환자 키
# PatientWeight  환자 나이
# PatientSex     환자 성별
# status         입원/외래/퇴원 구분
# PrscDrLcsNo    처방의사번호 (앞두자리가 의사 경력과 관련있음)
# disease_code   질병 코드

# RX_code        질병 코드
# count          일 횟수
# day            처방일수
# Amount         1회 용량 [소아과 1일 용량 / 나머지 1회 용량]

# InTime         intime
# OutTime        outtime
# timediff       직전처방과의 시간차이(단일 처방번호에서도 여러번에 로그가 있음)
# 210112         검사정보 210112
# 210109         검사정보 210109
# 210114         검사정보 210114
# 210113         검사정보 210113
# 210115         검사정보 210115
# 210111         검사정보 210111
# 12010101       검사정보 12010101
# 210105         검사정보 210105

## 처방약 + 처방번호 정보
rxinfo = pd.merge(rxinfo, df   
        ,how='left'
        ,on =['Basename','Idx']
        ,validate = "many_to_one")

## 각 처방정보의 최종처방만 정상데이터로 가정
tmp = rxinfo.groupby("Basename").Idx.max().reset_index()
## 해당 파일들에 있는 처방정보들은 오처방이 의심되어 제거
error = ["20180111A46810671RSA"
        ,"20180112A39585335TRSA"
        ,"20180122A22648717RSA"
        ,"20180129A39515041RSA"
        ,"20180129A39445044RSA"
        ,"20180130A37613467RSA"
        ,"20180309A23853966RSA"
        ,"20180309A46851773RSA"
        ,"20180313A49558833TRSA"
        ,"20180327A32479798RSA"
        ,"20180410A49530152RSA"
        ,"20180420A23678197RSA"
        ,"20180711A46497235RSA"
        ,"20180917A32632177RSA"
        ,"20180918A39658037RSA"]
tmp = tmp[tmp.Basename.isin(error) ==False]

## 각 처방정보에 최종처방은 정상데이터
tmp = tmp.assign(normal = 0)

rxinfo = pd.merge(rxinfo, tmp
        ,how='left'
        ,on =['Basename','Idx']
        ,validate = "many_to_one")

# label = pd.read_csv(os.path.join(BACKUP, "label_frame.csv"))
# label = label.rename(columns={"drug_modiy":"Drugcode"})
# label.Idx = label.Idx -1

# rxinfo = pd.merge(rxinfo, label[["Basename","Idx","Drugcode","label"]]
#         ,how='left'
#         ,on =['Basename','Idx',"Drugcode"]
#         ,validate = "many_to_one")

## 저장
rxinfo.to_csv(os.path.join(BACKUP,'frame.csv'), index=False)


