import pandas as pd
from hw1_1 import process_data_contain_hs300, ex_vector_compute, ex_matrix_compute, cov_matrix_compute
from config import RISK_FREE_RATE, DATAPATH, EXPECTED_RETURN
import statsmodels.api as sm
import os
import numpy as np


def random_sample_stock():
    x_matrix_total_hs300 = process_data_contain_hs300(DATAPATH)
    x_matrix_sample = x_matrix_total_hs300[1:].sample(n=5, axis=1, random_state=1, replace=False)
    x_matrix_sample["Market"] = x_matrix_total_hs300[0]
    return x_matrix_sample


def cov_matrix_with_market(x_matrix):
    day_yield_matrix, ex_numpy_vector = ex_vector_compute(x_matrix)
    # print(day_yield_matrix)
    # print(ex_numpy_vector)
    ex_matrix = ex_matrix_compute(day_yield_matrix, ex_numpy_vector)
    x_ex_matrix = day_yield_matrix - ex_matrix
    cov_matrix_numpy = cov_matrix_compute(x_ex_matrix)
    return cov_matrix_numpy


if __name__ == '__main__':
    x_matrix_sample = random_sample_stock()
    cov_matrix = cov_matrix_with_market(x_matrix_sample)
    cov_between_stock_market, market_variance = cov_matrix[-1][:-1], cov_matrix[-1][-1]
    beta_of_five_stocks = cov_between_stock_market / market_variance
    # 做OLS回归检验alpha（GRS检验是要分组的）
    _, day_avg_yield = ex_vector_compute(x_matrix_sample)
    annual_rate = day_avg_yield * x_matrix_sample.shape[0] / 10
    y = annual_rate[:-1] - RISK_FREE_RATE
    x = beta_of_five_stocks * (annual_rate[-1] - RISK_FREE_RATE)
    # x = np.ones(beta_of_five_stocks.shape) * (annual_rate[-1] - RISK_FREE_RATE)
    # print(x)

    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    filename = os.path.join(os.getcwd(), 'alpha_test')
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open("./alpha_test/alpha_test.txt", 'w') as f:
        f.write(str(results.params))
        f.write(str(results.summary()))
    print(results.params)
    print(results.summary())
