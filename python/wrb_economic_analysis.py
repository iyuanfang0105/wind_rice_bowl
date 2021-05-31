# %%
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


quandl.ApiConfig.api_key = 'DPj8KPgB9znTWb6ze242'

# %%

d = quandl.get_table(datatable_code='EDIA/ECD')

# %%
d[d.country_code == 'US']