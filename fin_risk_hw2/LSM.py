"""
Created on Sat Sep 12 14:05:58 2018
@author: Amin Jellali
@Email: amin.jellali@esprit.tn
"""

import numpy as np
import time as time_clock
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from hw2_1 import MonteCarloHW2
from config import *

begin = time_clock.time()
# other inputs refer to config

timePeriod = 1

print('Generating stock price matrix ...')

np.random.seed(1234)
mc_hw2 = MonteCarloHW2
stock_price, _, _ = mc_hw2.generate_samples()
b = np.array([[30] * 10000])
stock_price = np.insert(stock_price, 0, values=b, axis=0)

print('Calculating cash flow matrix...')
# At time t = N
cash_flow_matrix = np.zeros_like(stock_price)
for time in range(0, TRADING_DAYS + 1):
    for path_in_time in range(0, REPLICATIONS):
        if K - stock_price[time][path_in_time] > 0:
            cash_flow_matrix[time][path_in_time] = (
                    K - stock_price[time][path_in_time])
print('Entering loop...')
# starting the loop
for time in range(TRADING_DAYS - 1, 0, -1):
    print('Remaining calculations: ', time)
    # fetch the last cash flow for in the money paths
    X = []
    Y = []
    for stock_p in range(0, REPLICATIONS):
        if stock_price[time][stock_p] < K:
            X.append(stock_price[time][stock_p])
            for cash_flow_fetcher in range(time + 1, TRADING_DAYS + 1):
                if cash_flow_matrix[cash_flow_fetcher][stock_p] > 0:
                    Y.append(cash_flow_matrix[cash_flow_fetcher][stock_p] *
                             np.exp(-RISK_FREE_RATE_OPTION *
                                    DELTA_T * (cash_flow_fetcher - time)))
                    break
                elif (cash_flow_matrix[cash_flow_fetcher][stock_p] == 0 and
                      cash_flow_fetcher == TRADING_DAYS):
                    Y.append(0)
    # calculate the continuation values
    if len(X) > 1 and len(Y) > 1:
        poly = PolynomialFeatures(degree=ORDER_OF_POLYNOMIAL)
        x_poly = poly.fit_transform(np.array(X).reshape(-1, 1))
        # poly.fit(x_poly, y)
        lin = LinearRegression()
        lin.fit(x_poly, Y)
        continuation_values = lin.predict(x_poly)
        # continuation_values = renderContinuationValues(X, Y)
        # print(cash_flow_matrix)
        # generate the cash flow vector for time t
        # initialize the continuation values counter
        continuation_values_counter = -1
        # get the current cash_flow_vector of time
        cash_flow_vector = np.array([K - x
                                     if x < K else 0 for x in stock_price[time]])
        # set a dynamic comparision loop to compare between the cash flow at
        # current time and the corresponding continuation value
        for cash_flow_index in range(0, REPLICATIONS):
            if cash_flow_vector[cash_flow_index] > 0:
                continuation_values_counter += 1
                # have an in the money path thus we campare continuation values
                # to current vector
                if (cash_flow_vector[cash_flow_index] >
                        continuation_values[continuation_values_counter]):
                    # we exercice immedietly thus we change all future values
                    # to zero
                    for sub_time in range(time + 1, TRADING_DAYS + 1):
                        cash_flow_matrix[sub_time][cash_flow_index] = 0
                else:
                    cash_flow_matrix[time][cash_flow_index] = 0

# calculate mean values
cashFlowMeanVector = [np.mean(x) for x in cash_flow_matrix]
# discount mean values
DiscountedCashFlowVector = [cashFlowMeanVector[i] * np.exp(-RISK_FREE_RATE_OPTION * DELTA_T * i)
                            for i in range(1, len(cashFlowMeanVector))]
# determine the value
summ = np.sum(DiscountedCashFlowVector)
# determine best stoping time
maxCashFlow = np.max(DiscountedCashFlowVector)
end = time_clock.time()
standard_error_vector = np.matrix(stock_price)
standard_error = np.mean(standard_error_vector.std(1) / np.sqrt(REPLICATIONS))

with open("hw2_2_compare.txt", 'w') as f:
    f.write("finale value is: {}".format(summ) + '\n')
    f.write("execution time: {}".format(end - begin))

print('######################################################################')
print('############################ Input Data ##############################')
print('######################################################################')
print('          ', "number of paths is: ", REPLICATIONS)
print('          ', "time periode is: ", timePeriod)
print('          ', "number of exercice is: ", TRADING_DAYS)
print('          ', "step is : ", DELTA_T)
print('          ', "strike price is: ", K)
print('          ', "spot price is: ", S_0)
print('          ', "intrest rate is: ", RISK_FREE_RATE_OPTION)
print('          ', "volatility is: ", VOLATILITY)
print('          ', "number of polynoms is: ", ORDER_OF_POLYNOMIAL)
print('######################################################################')
print('########################### Final Values #############################')
print('######################################################################')
print('             ', " error is : ", standard_error)
print('          ', "max value is: ", maxCashFlow, " at time: ",
      DiscountedCashFlowVector.index(maxCashFlow))
print('          ', "finale value is: ", summ)
print('          ', "execution time: ", end - begin, ' s')
print('######################################################################')
print('######################################################################')
print('###############################AJ#####################################')
