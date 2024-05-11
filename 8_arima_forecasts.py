#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:08:27 2023

@author: yurifarod
"""

import time
import numpy as np
import pandas as pd
from pandas import Series
from statistics import mean

#Libs com os modelos de previsao
import statsmodels.api as sm

#Ignorar as Warnings do Codigo
import warnings
warnings.filterwarnings("ignore")

def previsao_arima(desp_serie, intervalo, parametros):

    measurement_error = parametros[0]
    time_varying_regression = parametros[1]
    enforce_stationarity = parametros[2]
    mle_regression = False
    concentrate_scale = parametros[3]
    cov_type = parametros[4]
    method = parametros[5]
    p = parametros[6]
    d = parametros[7]
    q = parametros[8]
    trend_offset = parametros[9]
    #prefixados
    order = (p,d,q)
    #Removendo a sazonalidade!
    seasonal_order = (0,0,0,0)
    arr_exog = None

    arr_real =[]
    inter_real = []
    for i in range(intervalo):
        max_size = max(desp_serie.index)
        arr_real.append(desp_serie[max_size])
        inter_real.append(max_size)
        desp_serie = desp_serie.drop(max_size)

    model = sm.tsa.statespace.SARIMAX(desp_serie,
                                      exog=arr_exog,
                                      order=order,
                                      seasonal_order=seasonal_order,
                                      mle_regression=mle_regression,
                                      hamilton_representation=False,
                                      simple_differencing=False,
                                      measurement_error=measurement_error,
                                      time_varying_regression=time_varying_regression,
                                      enforce_stationarity=enforce_stationarity,
                                      enforce_invertibility=False,
                                      concentrate_scale=concentrate_scale,
                                      initialization='approximate_diffuse',
                                      trend_offset=trend_offset)
    model_fit = model.fit(disp=0,
                          cov_type=cov_type,
                          method=method)

    start = len(desp_serie)
    end = len(desp_serie)+intervalo-1

    previsto = model_fit.predict(start = start, end = end)
    serie_real = Series(arr_real, inter_real)
    #Invertendo a serie
    serie_real = serie_real.iloc[::-1]

    size = len(previsto)
    erro = []
    real_size = min(serie_real.index)
    
    for i in range(real_size, real_size + size -1):
        parc_erro = (previsto[i] - serie_real[i])/serie_real[i]
        erro.append(abs(parc_erro))
    
    erro_medio = np.mean(erro)
    return abs(erro_medio), previsto, model_fit

class_file = './dataset/arrecadacao_icms_sergipe.csv'
df = pd.read_csv(class_file, index_col=False, sep=';')

df['fad_vltotal'] = df['fad_vltotal'].astype(float)

serie_original = df.iloc[:,1]

serie_original = serie_original.reindex()


#Esta serie tem a ultima observacao incompleta, precisaremos tratar isso!
serie_original.drop(serie_original.tail(1).index,inplace=True) # drop last n rows

serie_size = len(serie_original)
intervalo = 12


'''
Aqui iniciamos a previsao utilizando o modelo ARIMA (sarima zerado na sazonalidade)
'''
measurement_error = True
time_varying = True
enforce = True
concentrate = False
conv_type = 'approx'
method = 'cg'
p = 3
d = 1
q = 2
trend = 1

best_vlerro = 99
best_para = []
best_prev = []
testes = []


for test in range(14, -1, -1):
    serie_test = serie_original.copy()
    
    serie_test.drop(serie_test.tail(test).index,inplace=True)
    
    parametros = [measurement_error, time_varying, enforce, concentrate, conv_type, method, p, d, q, trend]
    vlerro, serie_prevista, modelo = previsao_arima(serie_test, intervalo, parametros)
    
    testes.append(round(vlerro*100, 2))

time.sleep(2)
print('======= Resultados ==========')
print(testes)
print('======= Média ==========')
print(mean(testes))

