import time

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from config import *
from hw2_1 import MonteCarloHW2


def longstaff_schwartz(stock_matrix):
    '''
    Calculate the option price of American put option by Longstaff_Schwartz method
    Valuing American Options by Simulation: A Simple Least-Squares Approach (Longstaff_Schwartz, 2001)
    :param stock_matrix: see method generate_samples
    :return: v_0: the option price of American put option
            v_m_matrix: the record of every step of the V_m
    '''
    cash_flow_matrix = np.maximum(0, K - stock_matrix)
    for i in range(TRADING_DAYS - 1, 0, -1):
        option_in_the_money_index = np.where(cash_flow_matrix[i - 1] != 0)[0].tolist()
        option_out_of_money_index = np.where(cash_flow_matrix[i - 1] == 0)[0].tolist()
        y = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * cash_flow_matrix[i][option_in_the_money_index]
        poly = PolynomialFeatures(degree=ORDER_OF_POLYNOMIAL)
        x_poly = poly.fit_transform(stock_matrix[i - 1][option_in_the_money_index].reshape(-1, 1))
        lin = LinearRegression()
        lin.fit(x_poly, y)
        continual_value = lin.predict(x_poly)

        # the result of polynomial regression is the same by using np.polyfit as follow:

        # z1 = np.polyfit(stock_matrix[i - 1][option_in_the_money_index], y, ORDER_OF_POLYNOMIAL)
        # p1 = np.poly1d(z1)
        # continual_value = p1(stock_matrix[i - 1][option_in_the_money_index])
        # print(continual_value)

        option_in_the_money_temp = cash_flow_matrix[i - 1][option_in_the_money_index]
        continual_index = np.where(cash_flow_matrix[i - 1][option_in_the_money_index] < continual_value)[0].tolist()
        option_in_the_money_temp[continual_index] = np.exp(
            -RISK_FREE_RATE_OPTION * DELTA_T) * cash_flow_matrix[i][option_in_the_money_index][continual_index]
        cash_flow_matrix[i - 1][option_in_the_money_index] = option_in_the_money_temp

        cash_flow_matrix[i - 1][option_out_of_money_index] = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * \
                                                             cash_flow_matrix[i][option_out_of_money_index]

    v_0 = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * np.mean(cash_flow_matrix[0])
    return v_0, cash_flow_matrix


def longstaff_schwartz_combine_tsitsiklis(stock_matrix):
    '''
    Combine of (Longstaff_Schwartz, 2001) and Tsitsiklis et al, 1999)
    :param stock_matrix: see method generate_samples
    :return: v_0: the option price of American put option
            v_m_matrix: the record of every step of the V_m
    '''
    cash_flow_matrix = np.maximum(0, K - stock_matrix)
    for i in range(TRADING_DAYS - 1, 0, -1):
        option_in_the_money_index = np.where(cash_flow_matrix[i - 1] != 0)[0].tolist()
        option_out_of_money_index = np.where(cash_flow_matrix[i - 1] == 0)[0].tolist()
        y = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * cash_flow_matrix[i][option_in_the_money_index]
        poly = PolynomialFeatures(degree=ORDER_OF_POLYNOMIAL)
        x_poly = poly.fit_transform(stock_matrix[i - 1][option_in_the_money_index].reshape(-1, 1))
        lin = LinearRegression()
        lin.fit(x_poly, y)
        continual_value = lin.predict(x_poly)

        # the result of polynomial regression is the same by using np.polyfit as follow:

        # z1 = np.polyfit(stock_matrix[i - 1][option_in_the_money_index], y, ORDER_OF_POLYNOMIAL)
        # p1 = np.poly1d(z1)
        # continual_value = p1(stock_matrix[i - 1][option_in_the_money_index])
        # print(continual_value)

        cash_flow_matrix[i - 1][option_in_the_money_index] = np.maximum(continual_value,
                                                                        cash_flow_matrix[i - 1][
                                                                            option_in_the_money_index])
        cash_flow_matrix[i - 1][option_out_of_money_index] = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * \
                                                             cash_flow_matrix[i][option_out_of_money_index]

    v_0 = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * np.mean(cash_flow_matrix[0])
    return v_0, cash_flow_matrix


def tsitsiklis(stock_matrix):
    '''
    Calculate the option price of American put option by Tsitsiklis et al.(1999) or the lecturenote5 from Prof. L. Jeff Hong method
    :param stock_matrix: see method generate_samples
    :return: v_0: the option price of American put option
            v_m_matrix: the record of every step of the V_m
    '''
    v_m_matrix = np.zeros(shape=(TRADING_DAYS, REPLICATIONS))
    v_m_vector = np.maximum(0, K - stock_matrix[TRADING_DAYS - 1])
    v_m_matrix[TRADING_DAYS - 1] = v_m_vector
    for i in range(TRADING_DAYS - 1, 0, -1):
        y = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * v_m_vector
        poly = PolynomialFeatures(degree=ORDER_OF_POLYNOMIAL)
        x_poly = poly.fit_transform(stock_matrix[i - 1].reshape(-1, 1))
        lin = LinearRegression()
        lin.fit(x_poly, y)
        c_x = lin.predict(x_poly)
        v_m_vector = np.maximum(c_x, np.maximum(K - stock_matrix[i - 1], 0))
        v_m_matrix[i - 1] = v_m_vector

    v_0 = np.exp(-RISK_FREE_RATE_OPTION * DELTA_T) * np.mean(v_m_vector)
    return v_0, v_m_matrix


if __name__ == '__main__':
    np.random.seed(1234)
    mc_hw2 = MonteCarloHW2
    stock_matrix, _, _ = mc_hw2.generate_samples()

    start = time.perf_counter()
    price_longstaff_schwartz, V_m_longstaff_schwartz = longstaff_schwartz(stock_matrix)
    end = time.perf_counter()
    print("Longstaff_Schwartz(2001)计算用时：{}".format(end - start))
    price_ls_t, V_m_ls_t = longstaff_schwartz_combine_tsitsiklis(stock_matrix)
    price_t, V_m_t = tsitsiklis(stock_matrix)

    with open("./hw2_2.txt", 'w') as f:
        f.write("Longstaff_Schwartz(2001)计算用时：{}".format(end - start) + '\n')
        f.write("Longstaff_Schwartz(2001) price : {}".format(price_longstaff_schwartz) + '\n')
        f.write(str(V_m_longstaff_schwartz) + '\n')
        f.write("Combine of (Longstaff_Schwartz, 2001) and Tsitsiklis et al, 1999) price: {}".format(price_ls_t) + '\n')
        f.write(str(V_m_ls_t) + '\n')
        f.write("Tsitsiklis et al.(1999) or the lecturenote5 from Prof. L. Jeff Hong price: {}".format(price_t) + '\n')
        f.write(str(V_m_ls_t) + '\n')
    print("Longstaff_Schwartz(2001) price : {}".format(price_longstaff_schwartz))
    print("Combine of (Longstaff_Schwartz, 2001) and Tsitsiklis et al, 1999) price: {}".format(price_ls_t))
    print("Tsitsiklis et al.(1999) or the lecturenote5 from Prof. L. Jeff Hong price: {}".format(price_t))
