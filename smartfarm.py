#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 12:14:58 2022

@author: jhy
"""

##Pycaret 설치

 
import pycaret
from pycaret.regression import *
import pandas as pd

 

##데이터 불러오기 및 검증 데이터 분리

 

smart_farm=pd.read_csv("/home/jhy/smartfarm/smart_farm_new.csv", index_col = 0)
smart_farm.tail(10) ##전체 데이터의 뒤에서 10개의 행만 보기 
smart_farm.shape

  

list(smart_farm.columns) ## 열 이름
nan=smart_farm[smart_farm['smart_farm.heat_supply'].isnull()] ##target 값이 nan인 행 확인
nan
nan.shape

 

smart_farm_train=smart_farm.dropna(subset=['smart_farm.heat_supply']) ##target 값이 nan이 아닐때만 가져옴
smart_farm_train.shape
smart_farm_train.info() ##nan 확인

 

##Setup

 
sup = setup(smart_farm_train, target = 'smart_farm.heat_supply', train_size = 0.6, ignore_features = [ 'smart_farm.yy'],fold_shuffle=True, session_id=2,silent=True)

##train set을 0.6으로 설정

##ignore feautures에서 학습하지 않을 열을 지정. 일단 년도만 제거

##fold shuffle은 k-fold cross 검증 시 shuffle 여부

##session id를 고정하면 모델을 고정할 수 있음  

 

models() ###사용가능한 모델 확인
best = compare_models(sort = 'RMSE',n_select=3) ##검증지수를 RMSE로 설정. 가장 성능좋은 모델 3개만 선택
best ##3개 보여줌

 

rf = create_model('rf', cross_validation = False) ##각각 model 생성, 생성시에는 모델이름 약자를 써야함, fold 사용해볼 예정
et = create_model('et', cross_validation = False)
lightgbm = create_model('lightgbm', cross_validation = False)

 

tuned_rf = tune_model(rf, optimize = 'RMSE',fold=2) ##생성된 모델에 대하여 튜닝. fold 개수 지정
tuned_et = tune_model(et, optimize = 'RMSE',fold=2)
tuned_li = tune_model(lightgbm, optimize = 'RMSE',fold=2)

 

blender_specific = blend_models(estimator_list = [tuned_rf,tuned_et,tuned_li], optimize = 'RMSE') #3가지 모델 앙상블하는 과정

 

 

###아래 코드 사용시 가장 best 모델 선정해서 튜닝가능하지만 시간이 너무 오래 걸려서 따로따로 모델 돌림 
#tuned_model = [tune_model(i) for i in best] ##너무 시간 오래 걸림
#blended = blend_models(tuned_model, optimize = 'RMSE')##위 코드 사용시 앙상블 할 때 사용가능

 

##간단한 분석

 

plot_model(tuned_rf, 'feature') ##importance feature selection
plot_model(blender_specific)#모델 설명력 확인 가능
plot_model(blender_specific, plot='error')
plot_model(blender_specific, plot='learning')

 

##최종 모델

final = finalize_model(blender_specific) ##k-fold로 나눠서 학습 및 검증하던 부분을 전체 합쳐서 다시 학습.

save_model(final,"/content/drive/MyDrive/model/"+"blend_model2") ##최종 모델 저장

 

 

final=load_model("/content/drive/MyDrive/model/blend_model2")##저장된 모델 불러올 때 
pred_holdout = predict_model(final)##모델 평가 

pred_holdout

 

from pycaret.utils import check_metric

check_metric(pred_holdout['smart_farm.heat_supply'], pred_holdout['Label'], metric = 'RMSE')##추정된 값과 정답값 비교

 

##검증

 

nan2=nan.drop(['smart_farm.heat_supply'],axis='columns') ##nan file가져오기
prediction=predict_model(data=nan2, estimator=final)##final model로 검증
prediction## 추정된 값

 

result=pd.DataFrame(prediction.groupby(['smart_farm.yy', 'smart_farm.mm','smart_farm.dd' ])['Label'].mean())##일별 평균으로 만들기 
result

 
result2=result.reset_index() ##데이터프레임 형식으로 만들기
result2

 

result3=result2[(result2['smart_farm.yy']==2022) & (result2['smart_farm.mm']==3)] #검증 데이터만 가져오려고 함. 하지만 검증하지 않을 날도 가져와짐.(3월 15일)
result3=result3.reset_index(drop=True) ##index 초기화

 

result3

 

result4=result3.drop(result3.index[10]) #3월 15일 제거
result4

 

result4.to_csv("/content/drive/MyDrive/smart farm/test.csv") ##제출할 파일 저장