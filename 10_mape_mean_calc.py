#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:04:48 2024

@author: yurifarod
"""

import numpy as np

mapes = np.genfromtxt('mapes_previsoes.csv', delimiter=',')
mapes = np.delete(mapes, 0, 0)

svr_mean = np.mean(mapes[0])
rnr_mean = np.mean(mapes[1])
arima_mean = np.mean(mapes[2])
ets_mapes = np.mean(mapes[3])