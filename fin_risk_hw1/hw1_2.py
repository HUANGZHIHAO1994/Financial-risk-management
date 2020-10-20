import pandas as pd
from hw1_1 import InvestmentStrategy as Is
from config import RISK_FREE_RATE, DATAPATH, EXPECTED_RETURN
import statsmodels.api as sm
import os
import numpy as np


def random_sample_stock():
    x_matrix_total_hs300 = Is.process_data_contain_hs300(DATAPATH)
    x_matrix_sample = x_matrix_total_hs300[1:].sample(n=5, axis=1, random_state=1, replace=False)
    x_matrix_sample["Market"] = x_matrix_total_hs300[0]
    return x_matrix_sample


def cov_matrix_with_market(x_matrix):
    day_yield_matrix, ex_numpy_vector = Is.ex_vector_compute(x_matrix)
    # print(day_yield_matrix)
    # print(ex_numpy_vector)
    ex_matrix = Is.ex_matrix_compute(day_yield_matrix, ex_numpy_vector)
    x_ex_matrix = day_yield_matrix - ex_matrix
    cov_matrix_numpy = Is.cov_matrix_compute(x_ex_matrix)
    return cov_matrix_numpy, day_yield_matrix


if __name__ == '__main__':
    x_matrix_sample = random_sample_stock()
    print(x_matrix_sample)
    cov_matrix, day_yield_matrix_market = cov_matrix_with_market(x_matrix_sample)
    cov_between_stock_market, market_variance = cov_matrix[-1][:-1], cov_matrix[-1][-1]
    beta_of_five_stocks = cov_between_stock_market / market_variance
    filename = os.path.join(os.getcwd(), 'beta')
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open("./beta/beta.txt", 'w') as f:
        f.write(str(beta_of_five_stocks))

    # # 做OLS回归检验alpha（GRS检验是要分组的）
    risk_free_rate_day = RISK_FREE_RATE / (x_matrix_sample.shape[0] / 10)
    R_i_R_m = day_yield_matrix_market.to_numpy() - risk_free_rate_day
    y = []
    beta_R_m = []
    R_m = day_yield_matrix_market.to_numpy().T[-1]
    for beta in beta_of_five_stocks:
        beta_R_m += (beta * R_m).tolist()

    for i in day_yield_matrix_market.to_numpy().T[:-1]:
        y += i.tolist()

    beta_R_m = sm.add_constant(beta_R_m)
    model = sm.OLS(y, beta_R_m)
    results = model.fit()
    filename = os.path.join(os.getcwd(), 'alpha_test')
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open("./alpha_test/alpha_test.txt", 'w') as f:
        f.write(str(results.params))
        f.write(str(results.summary()))
    print(results.params)
    print(results.summary())
