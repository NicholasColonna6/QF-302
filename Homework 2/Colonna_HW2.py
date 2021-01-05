
# coding: utf-8

# # QF 302 - Assignment 2
# ## Nicholas Colonna
# ### "I pledge my honor that I have abided by the Stevens Honor System." - ncolonna

# In[1]:


import shift
import sys
import time
import pandas as pd
import numpy as np
import statistics as stat
import statsmodels.api as sm
import statsmodels
import matplotlib.pyplot as plt


# In[3]:


# create trader object
trader = shift.Trader("ncolonna")


# In[4]:


# connect
trader.connect("initiator.cfg", "j7dR2Sad")


# In[5]:


# subscribe to all available order books
trader.subAllOrderBook()


# In[7]:


print(trader.getLastPrice('AAPL'))


# # Problem 1

# 1a) This is the main loop of our simulations. It performs 10 loops, saving the bid, ask and last price data for 120 periods every 5 seconds. It saves each simulation to csv file to be referenced later in the problem

# In[8]:


data = pd.DataFrame()
T=120

for i in range(1, 11):
    aapl = pd.DataFrame()
    
    for j in range(T):
        p = trader.getBestPrice('AAPL')
        aapl = aapl.append(pd.DataFrame({'BidPrice':p.getGlobalBidPrice(), 'BidSize':p.getGlobalBidSize(), 'AskPrice':p.getGlobalAskPrice(), 'AskSize':p.getGlobalAskSize(), 'Last':trader.getLastPrice("NFLX")}, index=[i]))
        time.sleep(5)
    
    aapl.to_csv('AAPL/aapl' + str(i) + '.csv', index=False)
    print('Status: ' + str(i) + ' of 10 loops completed')


# In[25]:


# disconnect
trader.disconnect()


# First, we will take a look at the first simulation.

# In[19]:


data = pd.read_csv('AAPL/aapl1.csv')
data.head()


# This section calculates the mid price, max and min of past n periods, as well as formulates the buy and sell signals of the algorithm. A delta of 0.2 resulted in the algorithm making 0 trades, therfore I divided by 1000, which was the threshold at which it started trading.

# In[111]:


n=10
delta=0.2 / 1000

data['Pk'] = (data['BidPrice'] + data['AskPrice']) / 2
data['Mk'] = data['Pk'].rolling(n).max().shift(1)
data['mk'] = data['Pk'].rolling(n).min().shift(1)
data['Buy'] = np.where((data['Pk']/data['Mk']) > (1+delta), 1, 0)
data['Sell'] = np.where((data['Pk']/data['mk']) < (1-delta), 1, 0)
data['signal'] = np.where(data['Buy'] == 1, 1, np.where(data['Sell'] == 1, -1, 0))


# In[112]:


data.tail()


# Here are 2 functions, one to calcuate the Profit and Loss of a given simulation, and another to calculate the number of trades made in a simulation.

# In[113]:


def calc_PnL(data):
    df2 = data[data['signal'] != 0]
    df2 = df2.append(data.iloc[-1])
    df2['ret'] = np.log(df2.AskPrice.shift(-1)) - np.log(df2.AskPrice)
    PnL = ((df2['signal'] * df2['ret']) + 1).prod()
    return PnL
    
def count_trades(data):
    num = 1;
    df2 = data[data['signal'] != 0]
    df2 = df2.append(data.iloc[-1])
    for i in range(1, len(df2['signal'])):
        if(df2['signal'].iloc[i-1] != df2['signal'].iloc[i]):
            num = num + 1
    return num


# We use the functions defined above to calculate the PnL and the Number of Trades in the first simulation

# In[114]:


pnl = calc_PnL(data)
print('Return of Sim1: ' + str(100*(pnl-1)) + "%")

num_trades = count_trades(data)
print('Number of Trades in Sim1: ' + str(num_trades))


# After going through the calculations above for the first simulation, I do a loop for the remaining simulations to perform the calculations and signals, and calculate PnL.

# In[170]:


PnLs = []
trades = []
PnLs.append(pnl)
trades.append(num_trades)

for i in range(2, 11):
    data = pd.read_csv('AAPL/aapl' + str(i) + '.csv')
    
    data['Pk'] = (data['BidPrice'] + data['AskPrice']) / 2
    data['Mk'] = data['Pk'].rolling(n).max().shift(1)
    data['mk'] = data['Pk'].rolling(n).min().shift(1)
    data['Buy'] = np.where((data['Pk']/data['Mk']) > (1+delta), 1, 0)
    data['Sell'] = np.where((data['Pk']/data['mk']) < (1-delta), 1, 0)
    data['signal'] = np.where(data['Buy'] == 1, 1, np.where(data['Sell'] == 1, -1, 0))

    
    pnl = calc_PnL(data)
    print("Return of Sim" + str(i) +": " + str(100*(pnl-1)) + "%")
    
    num_trades = count_trades(data)
    print("Number of Trades in Sim" + str(i) +": " + str(num_trades))
    
    PnLs.append(pnl)
    trades.append(num_trades)
    print()


# In[171]:


PnLs[:] = [100*(x - 1) for x in PnLs]
print("Average return across all simulations: " + str(stat.mean(PnLs)) + "%")
print("Average number of trades across all simulations: " + str(stat.mean(trades)))


# Below is a table with summary statistics on PnL and number of trades form the simulations

# In[172]:


stat_help = pd.DataFrame({'PnLs':PnLs, 'Num Trades':trades})
stat_help.describe()


# When running the code with delta is 0.2, the algorithm makes 0 trades during the time period, and therefore return is 0. I divided delta by 1000, which was a threshold for the algorithm to finally start making trades. For the next part we must test different values of delta to find a value that optimizes the PnL.

# 1b) As requested, I used a try and error approach to this problem where I tried various values of delta to see where I could get the greatest return. I kept multiplying the denominator by 10 to shrink the value of delta. Once I reached a value that seemed to keep PnL the same, I played with the numerator until I optimized my returns. The highest returns were achieved when delta = 0.00001 = 1/100000. This PnL is higher than the original version, as it made a profit rather than a loss.

# In[90]:


n=10
delta = 1/100000
PnLs = []
trades = []
PnLs.append(pnl)
trades.append(num_trades)

for i in range(1, 11):
    data = pd.read_csv('AAPL/aapl' + str(i) + '.csv')
    
    data['Pk'] = (data['BidPrice'] + data['AskPrice']) / 2
    data['Mk'] = data['Pk'].rolling(n).max().shift(1)
    data['mk'] = data['Pk'].rolling(n).min().shift(1)
    data['Buy'] = np.where((data['Pk']/data['Mk']) > (1+delta), 1, 0)
    data['Sell'] = np.where((data['Pk']/data['mk']) < (1-delta), 1, 0)
    data['signal'] = np.where(data['Buy'] == 1, 1, np.where(data['Sell'] == 1, -1, 0))

    pnl = calc_PnL(data)
    num_trades = count_trades(data)
    PnLs.append(pnl)
    trades.append(num_trades)

    
PnLs[:] = [x - 1 for x in PnLs]
print("Average return across all simulations: " + str(stat.mean(PnLs)) + "%")
print("Average number of trades across all simulations: " + str(stat.mean(trades)))


# # Problem 2

# First, I read in the S&P 500 data from the txt file. I then formated the date column to the correct format.

# In[3]:


sp500 = pd.read_csv('sp500hst.txt')
sp500['Date'] = pd.to_datetime(sp500['Date'], format='%Y%m%d').dt.strftime("%Y-%m-%d")
sp500.head()


# Next, I looped through the sp500 data and extracted only the data points for BAC, which will be used for this problem.

# In[4]:


bac = pd.DataFrame()
for i in range(0, len(sp500['Ticker'])):
    if sp500['Ticker'][i] == 'BAC':
        bac = bac.append(sp500.iloc[i])

bac = bac.reindex(sp500.columns, axis=1)
bac = bac.reset_index(drop=True)
bac.head()


# 2a) For MACD, I did the calculations for MACD and the Signal Line and added them to the data frame. Once completed, I was able to find the buy and sell signals buy checking the threshold for MACD crossing the signal line and from what direction

# In[5]:


bac['macd'] = bac['Close'].ewm(span=11, min_periods=11).mean() - bac['Close'].ewm(span=22, min_periods=22).mean()
bac['signal'] = bac['macd'].ewm(span=9, min_periods=9).mean()
bac['buy'] = np.where((bac['macd'].shift(1) < bac['signal'].shift(1)) & (bac['macd'] > bac['signal']), 1, 0)
bac['sell'] = np.where((bac['macd'].shift(1) > bac['signal'].shift(1)) & (bac['macd'] < bac['signal']), 1, 0)
bac.tail()


# Using the same method of calculating return from question 1, I calculated the PnL for the MACD algorithm. As you can see, it performed poorly over the specified time period.

# In[15]:


bac['bs_signal'] = np.where(bac['buy'] == 1, 1, np.where(bac['sell'] == 1, -1, 0))
BAC = bac[bac['bs_signal'] != 0]
BAC = BAC.append(bac.iloc[-1])
BAC['ret'] = np.log(BAC.Close.shift(-1)) - np.log(BAC.Close)
bac_pnl = ((BAC['bs_signal'] * BAC['ret']) + 1).prod()
print('Accumulative Return: ' + str(100*(bac_pnl-1)) + "%")


# The following is a table of all the buy and sell signals, the date of trade and return of that trade

# In[7]:


print(BAC[['Date', 'bs_signal', 'ret']])


# Lastly, I plotted the results of MACD and the Signal Line. MACD is in blue and Signal Line is in red. The green points on the plot represent each time a trade was made during the algorithms run period, which also signfies everywhere the signal line and macd intersect.

# In[14]:


fig, ax = plt.subplots()
plt.xlabel("Date")
plt.ylabel("EMA")
plt.title("MACD & Signal Line")
plt.plot(bac['Date'][29:],bac['macd'][29:])
plt.plot(bac['Date'][29:], bac['signal'][29:], color='red')
plt.scatter(BAC['Date'], BAC['signal'], marker='o', color='green')
plt.show()


# 2b) First, I read in the BAC data again to a new dataframe, and made the same adjustments as part a.

# In[20]:


bac2 = pd.DataFrame()
for i in range(0, len(sp500['Ticker'])):
    if sp500['Ticker'][i] == 'BAC':
        bac2 = bac2.append(sp500.iloc[i])

bac2 = bac2.reindex(sp500.columns, axis=1)
bac2 = bac2.reset_index(drop=True)
bac2.head()


# Next, I used the equations/algorithm defined on Canvas to create the RSI strategy. This includes finding where the price went up or down, and counting the number of up ticks and number of down ticks on a rolling sum basis. Next, I was able to calculate the RS ratio, and in turn, calculate the RSI. From there, I was able to determine the buy and sell signals using the overbought and oversold thresholds specified.

# In[21]:


N=14
ob = 80
os = 20

bac2['u'] = np.where(bac2['Close'] > bac2['Close'].shift(1), 1, 0)
bac2['d'] = np.where(bac2['Close'] < bac2['Close'].shift(1), 1, 0)
bac2['nup'] = bac2['u'].rolling(N).sum()
bac2['ndown'] = bac2['d'].rolling(N).sum()
bac2['n_up'] = bac2['nup'].ewm(span=N, min_periods=N).mean()
bac2['n_down'] = bac2['ndown'].ewm(span=N, min_periods=N).mean()
bac2['rs'] = bac2['n_up'] / bac2['n_down']
bac2['rsi'] = 100 * bac2['rs'] / (1 + bac2['rs'])
bac2['bs_signals'] = np.where((bac2['rsi'].shift(1) > os) & (bac2['rsi'] <= os), 1, np.where((bac2['rsi'].shift(1) < ob) & (bac2['rsi'] >= ob), -1, 0))

bac2.tail()


# Keeping it uniform throughout the code, I used the same method of calculating return as earlier. As you can see, RSI was able to generate a nice return.

# In[22]:


BAC2 = bac2[bac2['bs_signals'] != 0]
BAC2 = BAC2.append(bac2.iloc[-1])
BAC2['ret'] = np.log(BAC2.Close.shift(-1)) - np.log(BAC2.Close)
bac2_pnl = ((BAC2['bs_signals'] * BAC2['ret']) + 1).prod()
print('Accumulative Return: ' + str(100*(bac2_pnl-1)) + "%")


# The following is a table of all the buy and sell signals, the date of trade and return.

# In[23]:


print(BAC2[['Date', 'bs_signals', 'ret']])


# Finally, I plotted the RSI with the overbought and oversold lines. Whenever the RSI exceeds the red line, the algorithm will sell. Whenever the RSI falls below the green line, the algorithm will buy. The black dots represent where a trade was made.

# In[24]:


fig, ax = plt.subplots()
plt.xlabel("Date")
plt.ylabel("RSI")
plt.title("Relative Strength Index")
plt.plot(bac2['Date'][13:], bac2['rsi'][13:])
plt.hlines(ob, bac2['Date'][13], bac2['Date'].iloc[-1], color='red')
plt.hlines(os, bac2['Date'][13], bac2['Date'].iloc[-1], color='green')
plt.scatter(BAC2['Date'], BAC2['rsi'], marker='o', color='black')
plt.show()


# # Problem 3 

# First, I read in the data for the stocks XOM and CVX

# In[26]:


xom = pd.read_csv('XOM.csv')
cvx = pd.read_csv('CVX.csv')

xom.head()


# 3a) Next, I generated the log returns of XOM and CVX

# In[27]:


x = np.log(xom.Close) - np.log(xom.Close.shift(1))
y = np.log(cvx.Close) - np.log(cvx.Close.shift(1))
x = x.dropna()
y = y.dropna()


# From there, I ran a linear regression on the returns to estimate the co-integrating relation.

# In[28]:


mod = sm.OLS(y, x).fit()
print(mod.summary())


# As you can see below, here I calculate and plot of the residual. We will need this for the next part.

# In[29]:


residual = (mod.predict(x) - y)
plt.plot(residual)
plt.show()


# 3b) Next, I tested for stationarity of the residual with an Augmented Dickey Fuller unit root test using the package specified

# In[30]:


statsmodels.tsa.stattools.adfuller(residual)


# 3c) The next part I did was calculate delta using the formula given, which enabled me to calculate the buy and sell signals using the OLS coefficient, log returns of our stocks, and delta as the threshold.

# In[31]:


delta = 2 * np.std(residual)
signal = np.where(np.abs(y - 0.8882 * x) > delta, np.where((y - 0.8882 * x ) > delta, 1, -1), 0)


# Here, I appended those buy and sell signals to the xom and cvx dataframes, which will be needed for future calculation

# In[32]:


xom['signal'] = 0 * xom['Close']
cvx['signal'] = 0 * cvx['Close']

for i in range(0, len(xom['Close'])):
    xom['signal'][i] = signal[i-1]
    cvx['signal'][i] = -signal[i-1]


# Next, I created a new dataframe for xom and cvx, which I filtered only the entries where a trade is made using the signal

# In[33]:


XOM = xom[xom['signal'] != 0]
XOM = XOM.append(xom.iloc[-1])

CVX = cvx[cvx['signal'] != 0]
CVX = CVX.append(cvx.iloc[-1])

XOM


# Next, I calculate the returns from each of those trades

# In[34]:


XOM['ret'] = np.log(XOM.Close.shift(-1)) - np.log(XOM.Close)
CVX['ret'] = np.log(CVX.Close.shift(-1)) - np.log(CVX.Close)


# This is a table of XOM's long short shignals as well as returns for each trade

# In[35]:


XOM


# This is a table of CVX's long short shignals as well as returns for each trade

# In[36]:


CVX


# Finally, I calculate the individual accumulative returns for XOM and CVX, and then combine them to get the overall accumulative return

# In[37]:


retX = ((XOM['signal'] * XOM['ret']) + 1).prod() 
print('retX: ' + str(retX-1))

retY = ((CVX['signal'] * CVX['ret']) + 1).prod() 
print('retY: ' + str(retY-1))


# In[38]:


total_ret = retX * retY
print("Accumulative Return: " + str(100*(total_ret-1)) + "%")

