
# coding: utf-8

# # QF 302 - Assignment 1
# ## Nicholas Colonna
# ### "I pledge my honor that I have abided by the Stevens Honor System." - ncolonna

# In[1]:


import shift
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# create trader object
trader = shift.Trader("ncolonna")


# In[3]:


# connect
trader.connect("initiator.cfg", "j7dR2Sad")


# In[4]:


# subscribe to all available order books
trader.subAllOrderBook()


# # Problem 1

# In[5]:


bp = trader.getBestPrice('NFLX')


# In[6]:


print('NFLX', '\nBid:', bp.getBidPrice(), '\nAsk:', bp.getAskPrice())


# This loop will extract the Netflix bid price, bid size, ask price, ask size, and last price from SHIFT every 5 seconds for 10 minutes. This data will be used for further analysis in this assignment.

# In[7]:


#First loop for problem b 
#Results used later in the problem for printing QS, AS, and ILL and plotting
bp = trader.getBestPrice("NFLX")
T = 120

BA_data = pd.DataFrame()
for i in range(T):
    bp = trader.getBestPrice("NFLX")
    BA_data = BA_data.append(pd.DataFrame({'BidPrice':bp.getGlobalBidPrice(), 'BidSize':bp.getGlobalBidSize(), 'AskPrice':bp.getGlobalAskPrice(), 'AskSize':bp.getGlobalAskSize(), 'Last':trader.getLastPrice("NFLX")}, index=[i]))
    time.sleep(5)


# a) After running SHIFT for 10 minutes and extracting the price data every 5 seconds, we received the results below. The first table gives summary statistics for the bid prices and ask prices. The table after that is a preview of the data frame created from the trading period.

# In[8]:


BA_data[['BidPrice', 'AskPrice']].describe()


# In[9]:


BA_data.head()


# b) Utilizing the data from the SHIFT simulation, I calculated the Quoted Spread (QS), Average Spread (AS), and Illiquidity (ILL) at each time period. I appended this data to the data frame I created earlier

# In[10]:


BA_data['LogRet'] = np.log(BA_data['Last']) - np.log(BA_data['Last'].shift(1))
BA_data['QS'] = BA_data['AskPrice'] - BA_data['BidPrice']
BA_data['AS'] = np.abs(2 * (np.where(BA_data['LogRet'] >= 0,-1,1)) * (BA_data['Last'] - (0.5 * (BA_data['AskPrice'] + BA_data['BidPrice']))))
BA_data['ILL'] = np.abs(BA_data['LogRet']) / np.where(BA_data['LogRet'] >= 0, BA_data['AskSize'].shift(1), BA_data['BidSize'].shift(1))

BA_data.head()


# Next, I calculated the single value result for QS, AS and ILL, which essentially averages each metric across all time periods.
# 
# After that, I created a plot to show how each of these metrics changed during the trading period. The left axis is for QS (blue) and AS (yellow), and the right axis is for ILL (red).

# In[11]:


print('Quoted Spread:', 1/T * np.sum(BA_data['QS']))
print('Average Spread:', 1/T * np.sum(BA_data['AS']))
print('Illiquidity:', 1/T * np.sum(BA_data['ILL']))

get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax1 = plt.subplots()
plt.xlabel("Observation")
plt.ylabel("Spread")
BA_data['QS'][1:].plot(ax=ax1, legend=True)
BA_data['AS'][1:].plot(ax=ax1, legend=True)
ax2 = ax1.twinx()
BA_data['ILL'].plot(ax=ax2, color='r')
plt.title("Quoted Spread, Avg. Spread & Illiquidity")
plt.ylabel("Liquidity (Red)")
plt.show()


# c) d) Utilizing the the python correlation function, I calculated the correlation matrices for QS & AS and QS & ILL. From there, I extracted the correlations and printed to the screen.

# In[12]:


#Pearson correlation QS and AS
corr1 = BA_data[['QS', 'AS']].corr(method='pearson')['QS']['AS']
print('Pearson correlation between QS an AS =', corr1)

#Pearson correlation QS and ILL
corr2 = BA_data[['QS', 'ILL']].corr(method='pearson')['QS']['ILL']
print('Pearson correlation between QS an ILL =', corr2)


# I created 2 arrays to hold the correlations requested for parts c and d. I immediately added the data from the first loop to the arrays, the following data from the 10 loops will be appended later.

# In[13]:


corr_QS_AS = []
corr_QS_ILL = []

corr_QS_AS.append(corr1)
corr_QS_ILL.append(corr2)


# Next, I ran 10 additional loops, as specified for parts c and d.
# - In these loops, I decided not to save a majority of the data after exiting the loop
# - Since we are only concerned with the correlation between Quoted Spread and Average Spread, and Quoted Spread and Illiquidity, I calculated the necessary data, then calculated those correlations and saved to an array.

# In[14]:


for i in range(0, 10):
    df = pd.DataFrame()
    for i in range(1, T):
        bp = trader.getBestPrice("NFLX")
        df = df.append(pd.DataFrame({'BidPrice':bp.getGlobalBidPrice(), 'BidSize':bp.getGlobalBidSize(), 'AskPrice':bp.getGlobalAskPrice(), 'AskSize':bp.getGlobalAskSize(), 'Last':trader.getLastPrice("NFLX")}, index=[i]))
        time.sleep(5)
    
    df['LogRet'] = np.log(df['Last']) - np.log(df['Last'].shift(1))
    df['QS'] = df['AskPrice'] - df['BidPrice']
    df['AS'] = np.abs(2 * (np.where(df['LogRet'] >= 0,-1,1)) * (df['Last'] - (0.5 * (df['AskPrice'] + df['BidPrice']))))
    df['ILL'] = np.abs(df['LogRet']) / np.where(df['LogRet'] >= 0, df['AskSize'].shift(1), df['BidSize'].shift(1))
    
    corr_QS_AS.append(df[['QS', 'AS']].corr(method='pearson')['QS']['AS'])
    corr_QS_ILL.append(df[['QS', 'ILL']].corr(method='pearson')['QS']['ILL'])


# As requested, the average correlation for all 11 loops (original + 10 additional) was calculated for the respective variables below.

# In[25]:


print('Pearson correlations between QS and AS:')
print(corr_QS_AS)
print('\nPearson correlations between QS and ILL:')
print(corr_QS_ILL)

print('\nThe average Pearson correlation between QS and AS is:', np.mean(corr_QS_AS))
print('The average Pearson correlation between QS and ILL is:', np.mean(corr_QS_ILL))


# e) Observations on relationship between QS & AS and relationship between QS & ILL
# 
# - It can be seen from the results above that Quoted Spread and Average Spread have a higher average correlation than the relationship between Quoted Spread and Illiquidity.
# - The average correlation between QS and ILL is very low, so there isn't much correlation between the two metrics.
# - Given that QS and AS use the bid-ask spread and not volume, it is understandable why their average correlation is larger than the other relationship.
# - The range of correlations between QS and AS vary a decent amount getting as high as 0.38 and as low as 0.12, even going negative at times.
# - The range of correlations between QS and ILL vary a lot as well, going as high as 0.28 and as low as 0.02, even negative for some values.

# In[16]:


# disconnect
trader.disconnect()


# # Problem 2

# The first step of this problem was importing the data for Netflix (NFLX), which I saved to a Data Frame

# In[10]:


nflx = pd.read_csv('NFLX-2013_2018.csv')
nflx.head()


# With n=22, I calculated realized volatility using the standard equation and also beta.

# In[11]:


n=22
nflx['LogRet'] = np.log(nflx['Adj Close']) - np.log(nflx['Adj Close'].shift(1))

vol = (((nflx['LogRet'][-(n+1):-1] - nflx['LogRet'][-(n+1):-1].mean()) ** 2).mean()) ** (1/2)
beta = 2 / (n + 1)

print('sigma =', vol)
print('beta =', beta)


# a) Random Walk forecast:
# - The random walk forecast at time t is equal to the realized volatility at time t-1, so I calculated using the realized volatility equation

# In[12]:


vol_rw = (((nflx['LogRet'][-(n+1):-1] - nflx['LogRet'][-(n+1):-1].mean()) ** 2).mean()) ** (1/2)
vol_rw


# b) Exponential Smoothing Average:
# - Using the given equation, I forecast the volatility. However, I need to calculate the forecast for volatility at time t-1 as well. To do this, I found all the the forecasts using a loop and averaged them to use in the final calculation.

# In[13]:


sigma_hat = [(((nflx['LogRet'][-(n+i):-i] - nflx['LogRet'][-(n+i):-i].mean()) ** 2).mean()) ** (1/2) for i in range(1, len(nflx['LogRet'][0:-1]))]
vol_EMA = (1 - beta) * vol + beta * np.mean(sigma_hat)
vol_EMA


# c) Exponentially Weighted Moving Average:
# - First, I used a loop to make an array of beta to the 1 through n powers.
# - Next, I found the realized volatility for all t-i periods for i equals 1 through n.
# - Finally, I put it all together in the final equation given, summing the product of the calculated betas and volatilities and dividing by the volatilities

# In[14]:


beta_for_ewma = np.array([(2/(n+1))**i for i in range(1, len(nflx['LogRet'][0:-1]))])
sigma_for_ewma = np.array([(((nflx['LogRet'][-(n+i):-i] - nflx['LogRet'][-(n+i):-i].mean()) ** 2).mean()) ** (1/2) for i in range(1,len(nflx['LogRet'][0:-1]))])

vol_EWMA = ((beta_for_ewma * sigma_for_ewma).sum()) / beta_for_ewma.sum()
vol_EWMA


# d) Comparison and Contrast:
# - Out of all of the methods, the EMA gave the lowest value for volatility, followed by EWMA, and then RW with the largest.
# - The RW method only accounts for the volatility at the previous step, while the EMA and EWMA take into account all previous volatilities.
# - EMA gives all past volatilities equal weight in calculation, while EWMA gives more weight to the volatilities that are most recent
# 
# Recommendation:
# - I would recommend using the Exponentially Weighted Moving Average for analysis because of the property that it gives more weight to the most recent volatilities. The stock can start acting differently over time, so it's important to consider the most recent values a little more than older ones.

# # Problem 3 

# First, I created the covariance matrix for the Pt-1 and Pt. From there, I was able to extract gamma0 and gamma1.

# In[15]:


covar = np.cov((nflx['Adj Close'].shift(1) - nflx['Adj Close'].shift(2))[2:], (nflx['Adj Close'] - nflx['Adj Close'].shift(1))[2:])
print(covar, '\n')
gamma0 = covar[1,1]
gamma1 = covar[0,1]

print('Gamma 0:', gamma0)
print('Gamma 1:', gamma1)


# a) Next, I worked on deriving the value gammaL for all L>=2. It can be seen that the value of gamma for all L>=2 is equal to 0. (SEE HANDWRITTEN DERIVATION IN Problem3a.pdf SUBMITTED ALONGSIDE)

# Then, I calculated the mean gammaL from 2 to 200. As you can see, the average value is very small, however, it is not 0. That is because the derivation that gammaL is 0 is theoretical and not always true in real scenarios.

# In[16]:


gammaL = [np.cov((nflx['Adj Close'] - nflx['Adj Close'].shift(1))[2+i:], (nflx['Adj Close'].shift(i) - nflx['Adj Close'].shift(i+1))[2+i:])[0][1] for i in range(2, 100)]
print(np.mean(gammaL))


# b) Using the Roll Model, I worked to calculate the bid-ask spread and volatility.
# 
# BidAsk
# - First, I calculated c using the equation: Gamma1 = -c^2
# - Next, I was able to calculat the bid ask spread, which is equal to 2*c
# 
# Volatility
# - The fudamental volatility was calculated by using the equation: Gamma0 = 2*c^2 + sigma^2

# In[19]:


c = np.sqrt(np.abs(gamma1))

bidask = 2 * c
fund_vol = np.sqrt(gamma0 - 2 * (c ** 2))

print('Bid-Ask Spread:', bidask)
print('Fundamental Volatility:', fund_vol)

