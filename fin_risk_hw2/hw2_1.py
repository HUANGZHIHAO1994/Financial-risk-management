import numpy as np
from scipy import stats
from config import *


class MonteCarloHW2:
    @staticmethod
    def generate_samples():
        '''
        stock_matrix: (252 * 10000) pathwise sample stock price under BSM framework
        d_st_d_sigma_matrix: (252 * 10000) the derivative of stock by sigma for calculate vega
        standard_normal_matrix: (252 * 10000) samples of standard normal distribution
        :return: stock_matrix, d_st_d_sigma_matrix, standard_normal_matrix
        '''
        stock_matrix = np.zeros(shape=(TRADING_DAYS, REPLICATIONS))
        standard_normal_matrix = np.random.standard_normal(size=(TRADING_DAYS, REPLICATIONS))

        sum_of_standard_normal_matrix = np.zeros(shape=(TRADING_DAYS, REPLICATIONS))
        d_st_d_sigma_matrix = np.zeros(shape=(TRADING_DAYS, REPLICATIONS))

        for index, row in enumerate(standard_normal_matrix):
            if index == 0:
                sum_of_standard_normal_matrix[index] = standard_normal_matrix[index]
            else:
                sum_of_standard_normal_matrix[index] = sum_of_standard_normal_matrix[index - 1] + \
                                                       standard_normal_matrix[
                                                           index]

        for i, rep in enumerate(standard_normal_matrix):
            if i == 0:
                stock_matrix[i] = S_0 * np.exp(
                    (RISK_FREE_RATE_OPTION - (np.power(VOLATILITY, 2)) / 2) * DELTA_T + VOLATILITY * np.sqrt(
                        DELTA_T) * rep)
            else:
                stock_matrix[i] = stock_matrix[i - 1] * np.exp(
                    (RISK_FREE_RATE_OPTION - (np.power(VOLATILITY, 2)) / 2) * DELTA_T + VOLATILITY * np.sqrt(
                        DELTA_T) * rep)

            # for vega
            d_st_d_sigma_matrix[i] = stock_matrix[i] * (
                    -VOLATILITY * DELTA_T * (i + 1) + np.sqrt(DELTA_T) * sum_of_standard_normal_matrix[i])

        return stock_matrix, d_st_d_sigma_matrix, standard_normal_matrix

    @staticmethod
    def calculate(stock_matrix, d_st_d_sigma_matrix, standard_normal_matrix):
        '''
        s_i_bar_vector: (10000,) is the arithmetic mean of stock price of every sample path
        :param stock_matrix: see method generate_samples
        :param d_st_d_sigma_matrix: see method generate_samples
        :param standard_normal_matrix: see method generate_samples
        :return: price, confidence interval, delta, vega, gamma
        '''
        # option price
        s_i_bar_vector = np.mean(stock_matrix, axis=0)
        c_i_vector = np.exp(-RISK_FREE_RATE_OPTION * EXPIRES_ANNUALIZE) * np.maximum(0, s_i_bar_vector - K)
        c_price = np.mean(c_i_vector)

        # confidence_interval
        s_square = np.var(c_i_vector, ddof=1)
        confidence_interval = [
            c_price - stats.t.isf(ALPHA_OF_CONFIDENCE_INTERVAL / 2, df=(REPLICATIONS - 1)) * np.sqrt(
                s_square / REPLICATIONS),
            c_price + stats.t.isf(ALPHA_OF_CONFIDENCE_INTERVAL / 2, df=(REPLICATIONS - 1)) * np.sqrt(
                s_square / REPLICATIONS)]

        # pathwise delta and vega
        pathwise_delta = np.mean(
            np.exp(-RISK_FREE_RATE_OPTION * EXPIRES_ANNUALIZE) * (
                    np.sign(s_i_bar_vector - K) + 1) / 2 * s_i_bar_vector / S_0)
        d_sbar_d_sigma__vector = np.mean(d_st_d_sigma_matrix, axis=0)
        pathwise_vega = np.mean(
            np.exp(-RISK_FREE_RATE_OPTION * EXPIRES_ANNUALIZE) * (
                    np.sign(s_i_bar_vector - K) + 1) / 2 * d_sbar_d_sigma__vector)

        # gamma
        pathwise_gamma = np.mean(
            np.exp(-RISK_FREE_RATE_OPTION * EXPIRES_ANNUALIZE) * np.maximum(0, s_i_bar_vector - K) * (
                    np.power(standard_normal_matrix[0], 2) - standard_normal_matrix[0] * VOLATILITY * np.sqrt(
                DELTA_T) - 1) / (
                    np.power(S_0 * VOLATILITY, 2) * DELTA_T))
        return c_price, confidence_interval, pathwise_delta, pathwise_vega, pathwise_gamma


if __name__ == '__main__':
    np.random.seed(1234)
    mc_hw2 = MonteCarloHW2
    stock_matrix, d_st_d_sigma_matrix, standard_normal_matrix = mc_hw2.generate_samples()
    price, confidence, delta, vega, gamma = mc_hw2.calculate(stock_matrix, d_st_d_sigma_matrix, standard_normal_matrix)
    with open("./hw2_1.txt", 'w') as f:
        f.write(
            'price: {}\nconfidence: {}\ndelta: {}\nvega: {}\ngamma: {}\n'.format(price, str(confidence), delta, vega,
                                                                                 gamma))
    print('price: {}\nconfidence: {}\ndelta: {}\nvega: {}\ngamma: {}\n'.format(price, str(confidence), delta, vega,
                                                                               gamma))
