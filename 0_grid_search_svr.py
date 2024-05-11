#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:08:27 2023

@author: yurifarod
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Libs com os modelos de previsao
from sklearn.svm import SVR

#Ignorar as Warnings do Codigo
import warnings
warnings.filterwarnings("ignore")

def previsao_svr(desp_serie, intervalo, parametros):
    
    kernel = parametros[0]
    gamma = parametros[1]
    shrinking = parametros[2]
    verbose = parametros[3]
    
    normalizador = MinMaxScaler(feature_range=(0,1))
    desp_serie = normalizador.fit_transform(desp_serie)
    
    new_series = []
    arr_real = []
    inter_real = []
    for i in desp_serie:
        new_series.append(i)
        
    for i in range(intervalo):
        max_size = len(new_series)-1
        arr_real.append(desp_serie[max_size])
        inter_real.append(max_size)
        new_series = np.delete(new_series, max_size, 0)
    
    tamanho = len(new_series) - intervalo + 1
    timesteps = tamanho
    
    
    train_data_timesteps=np.array([[j for j in new_series[i:i+timesteps]] for i in range(0,len(new_series)-timesteps+1)])[:,:,0]
    
    x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
    
    model = SVR(kernel=kernel,gamma=gamma, shrinking=shrinking, verbose=verbose,
                C=10, epsilon = 0.05)    
    model.fit(x_train, y_train[:,0])
    #
    y_train_pred = model.predict(x_train).reshape(-1,1)
    y_train_pred = normalizador.inverse_transform(y_train_pred)
    arr_real = normalizador.inverse_transform(arr_real)
    #Invertendo a serie
    arr_real = arr_real[::-1]
    
    size = len(arr_real)
    erro = []
    
    for i in range(size):
        parc_erro = (y_train_pred[i]-arr_real[i])/arr_real[i]
        erro.append(abs(parc_erro))
    
    erro_medio = np.mean(erro)
    return abs(erro_medio), y_train_pred

class_file = './dataset/arrecadacao_icms_sergipe.csv'
df = pd.read_csv(class_file, index_col=False, sep=';')

df['fad_vltotal'] = df['fad_vltotal'].astype(float)

serie_original = df.iloc[:,1]

serie_original = serie_original.reindex()


#Esta serie tem as duas ultimas observacoes incompletas, precisaremos tratar isso!
serie_original.drop(serie_original.tail(1).index,inplace=True) # drop last n rows

#Agora colocamos a serie no ponto inicial dos testes (14 observacoes) para o grid-search
serie_original.drop(serie_original.tail(14).index,inplace=True)

serie_original = serie_original.values
serie_original = pd.DataFrame(serie_original)

serie_size = len(serie_original)
intervalo = 12


'''
Aqui iniciamos a previsao utilizando o modelo SVR
'''
v_kernel = ['rbf', 'sigmoid', 'linear', 'poly']
v_gamma = ['scale', 'auto']
v_shrinke = [True, False]
v_verbose = [True, False]

best_vlerro = 99
best_para = []
best_prev = []

for kernel in v_kernel:
    for gamma in v_gamma:
        for shrinke in v_shrinke:
            for verbose in v_verbose:
                parametros = [kernel, gamma, shrinke, verbose]
                vlerro, serie_prevista = previsao_svr(serie_original, intervalo, parametros)
                
                if vlerro < best_vlerro:
                    best_vlerro = vlerro
                    best_param = parametros
                    best_prev = serie_prevista


print('============ Melhores Parametros ============')

print(best_param)

