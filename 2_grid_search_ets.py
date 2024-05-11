#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:08:27 2023

@author: yurifarod
"""

import numpy as np
import pandas as pd
from pandas import Series

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
    
    size_ref = len(desp_serie)
    
    for i in range(size_ref, size_ref + size):
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

#Agora colocamos a serie no ponto inicial dos testes (14 observacoes) para o grid-search
serie_original.drop(serie_original.tail(14).index,inplace=True)

serie_size = len(serie_original)
intervalo = 12

'''
Aqui iniciamos a previsao utilizando o modelo ETS
'''
v_erro = ['add', 'mul']
v_trend = ['add', 'mul']
v_seasonal = ['add', 'mul']
v_init = ['estimated', 'heuristic']
v_period = list(range(2, 20))

best_vlerro = 99
best_para = []
best_prev = []
modelo = ''
for erro in v_erro:
    for trend in v_trend:
        for seasonal in v_seasonal:
            for init in v_init:
                for period in v_period:
                    parametros = [erro, trend, seasonal, init, period]
                    vlerro, serie_prevista, modelo = previsao_ets(serie_original, intervalo, parametros)
                    
                    if vlerro < best_vlerro:
                        best_vlerro = vlerro
                        best_param = parametros
                        best_prev = serie_prevista

print('============ Melhores Parametros ============')

print(best_param)
