# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:26:57 2023

@author: yfdantas
"""

import numpy as np

mapes = np.genfromtxt('mapes_previsoes.csv', delimiter=',')

mapes = np.delete(mapes, 0, 0)
nr_forecasts = 15
nr_algorithms = 4


regra_01 = 0
mape_total_01 = 0
regra_02 = 0
mape_total_02 = 0
regra_03 = 0
mape_total_03 = 0
regra_04 = 0
mape_total_04 = 0
for i in range(5, nr_forecasts):
    
    menor_rg_01 = 99
    menor_rg_02 = 99
    menor_rg_03 = 99
    menor_rg_04 = 99
    menor_mape_01 = 0
    menor_mape_02 = 0
    menor_mape_03 = 0
    menor_mape_04 = 0
    menor_prev = 99
    menor_id = 0
    for j in range(0, nr_algorithms):
        print(mapes[j][i])
        if mapes[j][i] < menor_prev:
            menor_prev = mapes[j][i]
            menor_id = j
            
        #Regra 01
        if mapes[j][i-1] < menor_rg_01:
            menor_rg_01 = mapes[j][i-1]
            menor_id_01 = j
            menor_mape_01 = mapes[j][i]
            
        #Regra 02
        vl_rg_02 = (mapes[j][i-1] + mapes[j][i-2] + mapes[j][i-3] + mapes[j][i-4] + mapes[j][i-5])/5
        if vl_rg_02 < menor_rg_02:
            menor_rg_02 = vl_rg_02
            menor_id_02 = j
            menor_mape_02 = mapes[j][i]
            
        #Regra 03
        vl_rg_03 = (5*mapes[j][i-1] + 4*mapes[j][i-2] + 3*mapes[j][i-3] + 2*mapes[j][i-4] + mapes[j][i-5])/15
        if  vl_rg_03 < menor_rg_03:
            menor_rg_03 = vl_rg_03
            menor_id_03 = j
            menor_mape_03 = mapes[j][i]
            
        #Regra 04
        vl_rg_04 = (mapes[j][i-1] + mapes[j][i-2] + mapes[j][i-3])/3
        if  vl_rg_04 < menor_rg_04:
            menor_rg_04 = vl_rg_04
            menor_id_04 = j
            menor_mape_04 = mapes[j][i]
    
    mape_total_01 += menor_mape_01
    mape_total_02 += menor_mape_02
    mape_total_03 += menor_mape_03
    mape_total_04 += menor_mape_04
    
    print('=================================')
    print('Menor Previsao:  ' + str(menor_id))
    print('Avaliação, regra 01: ' + str(menor_id_01))
    print('Avaliação, regra 02: ' + str(menor_id_02))
    print('Avaliação, regra 03: ' + str(menor_id_03))
    print('Avaliação, regra 04: ' + str(menor_id_04))
    print('=================================')
    
    if menor_id == menor_id_01:
        regra_01 += 1
    if menor_id == menor_id_02:
        regra_02 += 1
    if menor_id == menor_id_03:
        regra_03 += 1
    if menor_id == menor_id_04:
        regra_04 += 1
        
print('============ Resultado ==========')
print('Regra 01: '+str(regra_01/(nr_forecasts-3) )+'; Mape Médio: '+str(mape_total_01/(nr_forecasts-3)))
print('Regra 02: '+str(regra_02/(nr_forecasts-3) )+'; Mape Médio: '+str(mape_total_02/(nr_forecasts-3)))
print('Regra 03: '+str(regra_03/(nr_forecasts-3) )+'; Mape Médio: '+str(mape_total_03/(nr_forecasts-3)))
print('Regra 04: '+str(regra_04/(nr_forecasts-3) )+'; Mape Médio: '+str(mape_total_04/(nr_forecasts-3)))