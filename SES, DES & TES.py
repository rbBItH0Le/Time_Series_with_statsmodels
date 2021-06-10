# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:42:45 2021

@author: rohan
"""

import pandas as pd
import numpy as np

dataset=pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
dataset.index.freq='MS'
dataset.index

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
span=12
alpha=2/(span+1)

dataset['EWMA12']=dataset['Thousands of Passengers'].ewm(alpha).mean()

model=SimpleExpSmoothing(dataset['Thousands of Passengers'])

fitted_model=model.fit(smoothing_level=alpha,optimized=False)
dataset['SES12']=fitted_model.fittedvalues.shift(-1)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

dataset['DES_MUL_12']=ExponentialSmoothing(dataset['Thousands of Passengers'],trend='mul').fit().fittedvalues.shift(-1)

dataset.plot()

dataset.iloc[-24:].plot()

dataset['TES_MUL_12']=ExponentialSmoothing(dataset['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
