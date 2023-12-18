#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:08:27 2023

@author: yurifarod
"""
import time
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.preprocessing import MinMaxScaler


#Ignorar as Warnings do Codigo
import warnings
warnings.filterwarnings("ignore")

class_file = './dataset/arrecadacao_icms_sergipe.csv'
df = pd.read_csv(class_file, index_col=False, sep=';')

df['fad_vltotal'] = df['fad_vltotal'].astype(float)

serie_original = df.iloc[:,1]

serie_original = serie_original.reindex()


#Esta serie tem as duas ultimas observacoes incompletas, precisaremos tratar isso!
serie_original.drop(serie_original.tail(1).index,inplace=True) # drop last n rows

serie_original = serie_original.values
serie_original = pd.DataFrame(serie_original)

serie_size = len(serie_original)
intervalo = 12


#Bibliotecas necessarias ao LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten

def previsao_lstm(desp_serie, intervalo, parametros):
    
    epochs = parametros[0]
    units = parametros[1]
    activation = parametros[2]
    final_activation = parametros[3]
    optimizer = parametros[4]

    normalizador = MinMaxScaler(feature_range=(0,1))
    base_treinamento_normalizada = normalizador.fit_transform(desp_serie)

    max_size = len(desp_serie)

    previsores = []
    preco_real = []
    for i in range(12, max_size - intervalo):
        previsores.append(base_treinamento_normalizada[i-12:i, 0])
        preco_real.append(base_treinamento_normalizada[i, 0])

    previsores = np.array(previsores)
    preco_real = np.array(preco_real)
    previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

    regressor = Sequential()
    regressor.add(LSTM(units = units, activation = activation, return_sequences=True,
                       recurrent_activation='sigmoid', input_shape=(previsores.shape[1], 1)))
    regressor.add(Dropout(0.3))

    regressor.add(LSTM(units = units, activation = activation, return_sequences=True,
                       recurrent_activation='sigmoid'))
    regressor.add(Dropout(0.3))

    regressor.add(LSTM(units = units, activation = activation, return_sequences=True, recurrent_activation='sigmoid'))
    regressor.add(Dropout(0.3))

    regressor.add(LSTM(units = units, activation = activation, return_sequences=True, recurrent_activation='sigmoid'))
    regressor.add(Dropout(0.3))

    regressor.add(Flatten())
    regressor.add(Dense(units = 1, activation = final_activation))

    regressor.compile(optimizer = optimizer,
                      loss = 'mean_squared_error',
                      metrics = ['mean_absolute_error'])

    regressor.fit(previsores, preco_real, epochs = epochs, batch_size = 1)

    base_teste = desp_serie[(max_size - intervalo):max_size]
    base_completa = desp_serie
    entradas = base_completa[len(base_completa) - len(base_teste) - 12:].values
    entradas = entradas.reshape(-1, 1)
    entradas = normalizador.transform(entradas)

    X_teste = []
    for i in range(12, 12 + intervalo):
        X_teste.append(entradas[(i-12):i, 0])
    X_teste = np.array(X_teste)
    X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

    previsoes = regressor.predict(X_teste)
    previsoes = normalizador.inverse_transform(previsoes)

    #Para plotagem somente
    base_teste = np.array(base_teste)

    size = len(previsoes)
    erro = []

    for i in range(size):
        parc_erro = (previsoes[i] - base_teste[i])/base_teste[i]
        erro.append(abs(parc_erro))

    erro_medio = np.mean(erro)
    return abs(erro_medio), previsoes

'''
Aqui iniciamos a previsao utilizando o modelo LSTM
'''
epochs = 20
units = 36
activation = 'softsign'
final_activation = 'selu'
optimizer = 'rmsprop'

best_vlerro = 99
best_para = []
best_prev = []

testes = []


for test in range(14, -1, -1):
    serie_test = serie_original.copy()
    
    serie_test.drop(serie_test.tail(test).index,inplace=True)
    
    parametros = [epochs, units, activation, final_activation, optimizer]
    vlerro, serie_prevista = previsao_lstm(serie_test, intervalo, parametros)
    
    testes.append(round(vlerro*100, 2))

time.sleep(2)
print('======= Resultados ==========')
print(testes)
print('======= Média ==========')
print(mean(testes))



