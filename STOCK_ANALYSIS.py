
# coding: utf-8

# In[ ]:


# importing Libraries
from nsepy import get_history
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


from datetime import date


# In[4]:


data = get_history(symbol="INFY", start=date(2015,1,1), end=date(2016,12,31))
data[['Close']].plot()


# In[5]:


infosys_data = get_history(symbol='INFY',
                   start=date(2015,1,1),  
                   end=date(2016,12,31))


# In[6]:


TCS_data = get_history(symbol='TCS',
                   start=date(2015,1,1),  
                   end=date(2016,12,31))


# In[7]:


print(infosys_data)


# In[8]:


print(TCS_data)


# In[9]:


infosys_data.head(9)


# In[10]:


TCS_data.head(9)


# In[11]:


import numpy as np
infosys_data_log = np.log(infosys_data['Close'])
plt.plot(infosys_data_log)


# In[12]:


import numpy as np
TCS_data_log = np.log(TCS_data['Close'])
plt.plot(TCS_data_log)


# In[13]:


# Moving average WINDOW=10
# INFOSYS
moving_avg = infosys_data_log.rolling(10).mean()
plt.plot(infosys_data_log)
plt.plot(moving_avg, color='red')


# In[14]:


# Moving average WINDOW=10
# TCS
moving_avg2 = TCS_data_log.rolling(10).mean()
plt.plot(TCS_data_log)
plt.plot(moving_avg2, color='red')


# In[15]:


# WINDOW = 75
moving_avg = infosys_data_log.rolling(75).mean()
plt.plot(infosys_data_log)
plt.plot(moving_avg, color='red')


# In[16]:


# WINDOW = 75
moving_avg = TCS_data_log.rolling(75).mean()
plt.plot(TCS_data_log)
plt.plot(moving_avg2, color='red')


# In[17]:


den_log_moving_avg_diff = infosys_data_log - moving_avg
den_log_moving_avg_diff.head(80)


# In[18]:


den_log_moving_avg_diff2 = TCS_data_log - moving_avg2
den_log_moving_avg_diff2.head(80)


# In[19]:


# Dropping Null values
den_log_moving_avg_diff.dropna(inplace=True)
den_log_moving_avg_diff2.dropna(inplace=True)


# In[20]:


# differencing
#INFOSYS
den_log_diff = infosys_data_log - infosys_data_log.shift()
plt.plot(den_log_diff)


# In[21]:


# differencing
#TCS
den_log_diff2 = TCS_data_log - TCS_data_log.shift()
plt.plot(den_log_diff2)


# In[22]:




# Calculate the 10 and 75 days moving averages of the closing prices
short_rolling_msft = infosys_data_log.rolling(window=10).mean()
long_rolling_msft = infosys_data_log.rolling(window=75).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(infosys_data_log.index, infosys_data_log, label='MSFT')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='10 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label='75 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()


# In[23]:


# Forecasting Time series


# In[24]:


from statsmodels.tsa.arima_model import ARIMA


# In[25]:


# AR model
model = ARIMA(infosys_data_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(den_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-den_log_diff)**2))


# In[26]:


# MA model

model = ARIMA(infosys_data_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(den_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-den_log_diff)**2))


# In[27]:


#Futures and Options historical data
nifty_fut = get_history(symbol="NIFTY", 
start=date(2015,1,1), 
end=date(2015,1,10),
index=True,
futures=True, expiry_date=date(2015,1,29))


# In[28]:


print(nifty_fut)

