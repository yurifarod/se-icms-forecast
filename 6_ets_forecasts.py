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
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

#Ignorar as Warnings do Codigo
import warnings
warnings.filterwarnings("ignore")

def previsao_ets(desp_serie, intervalo, parametros):

    error = parametros[0]
    trend = parametros[1]
    seasonal = parametros[2]
    initialization = parametros[3]
    seasonal_periods = parametros[4]

    arr_real =[]
    inter_real = []

    for i in range(intervalo):
        max_size = max(desp_serie.index)
        arr_real.append(desp_serie[max_size])
        inter_real.append(max_size)
        desp_serie = desp_serie.drop(max_size)

    model = ETSModel(desp_serie, error=error, trend=trend, seasonal=seasonal,
                     initialization_method=initialization, seasonal_periods=seasonal_periods)
    model_fit = model.fit(disp=0)

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
Aqui iniciamos a previsao utilizando o modelo ETS
'''
erro = 'mul'
trend = 'mul'
seasonal = 'mul'
init = 'heuristic'
period = 15

best_vlerro = 99
best_para = []
best_prev = []
testes = []


for test in range(14, -1, -1):
    serie_test = serie_original.copy()
    
    serie_test.drop(serie_test.tail(test).index,inplace=True)
    
    parametros = [erro, trend, seasonal, init, period]
    vlerro, serie_prevista, modelo = previsao_ets(serie_test, intervalo, parametros)
    
    testes.append(round(vlerro*100, 2))

time.sleep(2)
print('======= Resultados ==========')
print(testes)
print('======= Média ==========')
print(mean(testes))

