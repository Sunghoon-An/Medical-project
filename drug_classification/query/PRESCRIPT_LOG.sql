## ==========================================================================================================
## MODEL에서 필요한 정보
## ==========================================================================================================
## DRUG_CODE      : 약물 코드 
## DISEASE_CODE   : 질병코드 
## DOSAGE         : 용법코드
## AMOUNT         : 용량 
## COUNT          : 횟수 
## DAYS           : 처방일수 
## HEIGHT         : 환자 키 
## WEIGHT         : 환자 몸무게 
## SEX            : 환자 성별 
## PRSCDRLCSNO    : 처방의 면허 번호 
## LAB_1742-6     : 공통검사 코드 (1742-6) 검새내용 (간기능) 단위 (IU/L)    이름 (Alanine transaminase)
## LAB_1751-7     : 공통검사 코드 (1751-7) 검새내용 (간기능) 단위 (g/dL)    이름 (Albumin)
## LAB_1968-7     : 공통검사 코드 (1968-7) 검새내용 (간기능) 단위 (mg/dL)   이름 (Bilirubin| direct)
## LAB_1975-2     : 공통검사 코드 (1975-2) 검새내용 (간기능) 단위 (mg/dL)   이름 (Bilirubin| total)
## LAB_2324-2     : 공통검사 코드 (2324-2) 검새내용 (간기능) 단위 (IU/L)    이름 (Gamma-glutamyl transpeptidase)
## LAB_1920-8     : 공통검사 코드 (1920-8) 검새내용 (간기능) 단위 (IU/L)    이름 (Aspartate transaminase)
## LAB_5902-2     : 공통검사 코드 (5902-2) 검새내용 (간기능) 단위 (sec )    이름 (처방양)
## LAB_2160-0     : 공통검사 코드 (2160-0) 검새내용 (신장기능) 단위 (mg/dL) 이름 (Creatinine)
## AGE            : 나이(생년월)
## AMT_PER_WEIGHT : 몸무게당 용량
## TOTAL_AMT      : 총 용량
## label          : 

## ==========================================================================================================
## COMPARATOR SQL
## ==========================================================================================================
SELECT PRES.UUID AS UUID
       GDP.CODE AS DRUG_CODE,              -- Drugcode 약품코드
       DISEASE.CODE AS DISEASE_CODE,       -- disease_code 질병코드
       GDP.DOSAGE_CODE AS DOSAGE,          -- Dosage 병원 용법코드
       GDP.COUNT AS AMOUNT,                -- Amount 처방용량
       GDP.AMOUNT * GDP.COUNT AS COUNT,    -- Count 횟수
       GDP.DAYS AS DAYS,                   -- Days 처방일수
       PRES.HEIGHT AS HEIGHT,              -- PatientHeight 환자 키
       PRES.WEIGHT AS WEIGHT,              -- PatientWeight 환자 몸무게
       PRES.SEX AS SEX,                    -- PatientSex 환자 성별
       PRES.PRSC_DR_LCS_NO AS PRSCDRLCSNO, -- PrscDrLcsNo 처방의 번호
       PRES.LAB_1742_6 AS LAB_1742_6,      -- 검사정보
       PRES.LAB_1751_7 AS LAB_1751_7,  
       PRES.LAB_1968_7 AS LAB_1968_7,  
       PRES.LAB_1975_2 AS LAB_1975_2,  
       PRES.LAB_2324_2 AS LAB_2324_2,  
       PRES.LAB_1920_8 AS LAB_1920_8,  
       PRES.LAB_5902_2 AS LAB_5902_2,
       PRES.LAB_2160_0 AS LAB_2160_0,      
       PRES.AGE AS AGE,                    -- age 환자나이(월 단위 34개월) 
       GDP.COUNT * GDP.AMOUNT AS TOTAL_AMT -- total_amt 총 용량
       0 AS LABEL                          -- 향후 UPDATE 할것을 고려하여 생성
       
  FROM (
        SELECT MIN(GPP.UUID) AS UUID,
               MIN(DATEDIFF(MONTH , TO_DATE(SUBSTR(GPP.BIRTH,1,7),'YYYY-MM'), SYSDATE )) AS AGE,
               MIN(GPP.HEIGHT) AS HEIGHT,
               MIN(GPP.WEIGHT) AS WEIGHT,
               MIN(GPP.SEX) AS SEX,
               MIN(GPP.PRSC_DR_LCS_NO) AS PRSC_DR_LCS_NO,
               MAX(CASE WHEN LAB.CODE = '1742-6' THEN LAB.VALUE ELSE NULL END) AS LAB_1742_6,
               MAX(CASE WHEN LAB.CODE = '1751-7' THEN LAB.VALUE ELSE NULL END) AS LAB_1751_7,
               MAX(CASE WHEN LAB.CODE = '1968-7' THEN LAB.VALUE ELSE NULL END) AS LAB_1968_7,
               MAX(CASE WHEN LAB.CODE = '1975-2' THEN LAB.VALUE ELSE NULL END) AS LAB_1975_2,
               MAX(CASE WHEN LAB.CODE = '2324-2' THEN LAB.VALUE ELSE NULL END) AS LAB_2324_2,
               MAX(CASE WHEN LAB.CODE = '1920-8' THEN LAB.VALUE ELSE NULL END) AS LAB_1920_8,
               MAX(CASE WHEN LAB.CODE = '5902-2' THEN LAB.VALUE ELSE NULL END) AS LAB_5902_2,
               MAX(CASE WHEN LAB.CODE = '2160-0' THEN LAB.VALUE ELSE NULL END) AS LAB_2160_0,
               MIN(GPP.SUBJECT) AS SUBJECT,
               MIN(GPP.STATUS) AS STATUS
          FROM STG.GS_PATIENT_PRESCRIPT_LOG GPP,
               STG.GS_PATIENT_LABS_LOG LAB
         WHERE 1=1
           AND GPP.UUID = ?
           AND GPP.UUID = LAB.PARENT_UUID(+)
       ) AS PRES,
       STG.GS_DISEASE_PRESCRIPT_LOG DISEASE,
       STG.GS_DRUG_PRESCRIPT_LOG GDP
 WHERE 1=1 
   AND PRES.SUBJECT = 'PED ' -- [주의] 진료과 코드 3자리 이후 공백문자 하나있음
   AND PRES.STATUS = 'O'
   AND PRES.AGE <= 192
   AND PRES.UUID = GDP.PARENT_UUID
   AND PRES.UUID = DISEASE.PARENT_UUID
   AND DISEASE.CLASS = 'M';
