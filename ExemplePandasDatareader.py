# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:23:48 2021

@author: Robert
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd

symboles = ['AAPL', 'MSFT', '^GSPC'] # Symboles boursiers de Apple, Microsoft et indice S&P500

# We would like all available data from 01/01/2000 until 12/31/2016.
date_debut = '2010-01-01'
date_fin = '2016-12-31'

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader('WIKI/AAPL', 'quandl', date_debut, date_fin)

panel_data.to_frame().head(9)

# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data['Close']

# Getting all weekdays between 01/01/2000 and 12/31/2016
all_weekdays = pd.date_range(start=date_debut, end=date_fin, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

# Reindexing will insert missing values (NaN) for the dates that were not present
# in the original set. To cope with this, we can fill the missing by replacing them
# with the latest available price for each instrument.
close = close.fillna(method='ffill')

print(all_weekdays)

# DatetimeIndex(['2010-01-01', '2010-01-04', '2010-01-05', '2010-01-06',
#               '2010-01-07', '2010-01-08', '2010-01-11', '2010-01-12',
#               '2010-01-13', '2010-01-14',
#               ...
#               '2016-12-19', '2016-12-20', '2016-12-21', '2016-12-22',
#               '2016-12-23', '2016-12-26', '2016-12-27', '2016-12-28',
#               '2016-12-29', '2016-12-30'],
#              dtype='datetime64[ns]', length=1826, freq='B')

