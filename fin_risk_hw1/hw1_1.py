import matplotlib
import pandas as pd
import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from colorama import Fore
from config import RISK_FREE_RATE, DATAPATH, EXPECTED_RETURN, STOCKS_NUMBER, MONTO_CARLO_TIMES


class InvestmentStrategy:

    @staticmethod
    def process_data_x_matrix(datapath):
        df_raw = pd.read_excel(datapath)
        df_raw = df_raw.T
        df_raw = df_raw.drop(index=['code', 'name'], columns=[0])
        df_raw = df_raw.fillna(method='ffill')
        # 第32只股票第一天就是空缺值，用向前填补方式
        df = df_raw.fillna(method='backfill')
        return df

    @staticmethod
    def process_data_contain_hs300(datapath):
        df_raw = pd.read_excel(datapath)
        df_raw = df_raw.T
        df_raw = df_raw.drop(index=['code', 'name'])
        # df_raw.to_excel("./test1.xlsx")
        # print(df_raw.isnull().any())
        df_raw = df_raw.fillna(method='ffill')

        # 第32只股票第一天就是空缺值，用向前填补方式
        df = df_raw.fillna(method='backfill')
        return df

    @staticmethod
    def day_yield_compute(x_matrix):
        day_yield = (x_matrix.shift(-1) - x_matrix) / x_matrix
        return day_yield.iloc[:-1, :]

    @staticmethod
    def ex_vector_compute(x_matrix):
        day_yield = (x_matrix.shift(-1) - x_matrix) / x_matrix
        day_avg_yield = day_yield.mean().to_numpy()
        return day_yield.iloc[:-1, :], day_avg_yield

    @staticmethod
    def ex_matrix_compute(x_matrix, ex_numpy_vector):
        ex_np = np.repeat(np.expand_dims(ex_numpy_vector, axis=0), x_matrix.shape[0], axis=0)
        ex_matrix = pd.DataFrame(ex_np, index=x_matrix.index, columns=x_matrix.columns)
        return ex_matrix

    @staticmethod
    def cov_matrix_compute(x_ex_matrix):
        return np.matmul(x_ex_matrix.T.to_numpy(), x_ex_matrix.to_numpy()) / (x_ex_matrix.shape[0] - 1)

    def compute_weight(self, x_matrix, total_days=252, method="Markowitz", starttime=0, endtime=0):
        # ex_numpy_vector是r拔 (50,) numpy
        # x_matrix是矩阵X [6个月天数 rows x 50 columns] 比如第一次计算权重就是(1212, 50)
        # ex_matrix是EX矩阵 [6个月天数 rows x 50 columns]
        # x_ex_matrix矩阵X-EX
        # 协方差矩阵：cov (50, 50)
        total_days_every_year = total_days / 5

        day_yield_matrix, ex_numpy_vector = self.ex_vector_compute(x_matrix)
        ex_matrix = self.ex_matrix_compute(day_yield_matrix, ex_numpy_vector)
        x_ex_matrix = day_yield_matrix - ex_matrix
        cov_matrix_numpy = self.cov_matrix_compute(x_ex_matrix)

        # stocks_number = 50
        n = STOCKS_NUMBER

        one_matrix = np.ones((1, n))

        '''
        # cvxopt这个包也能做
        P = matrix(cov_matrix_numpy.tolist())
        # print(P)
        # print('*' * 50)
        q = matrix([0.0] * 50)
        # G = matrix([[-1.0, 0.0], [0.0, -1.0]])
        # h = matrix([0.0, 0.0])
        A = matrix(np.vstack((ex_numpy_vector, one_matrix)))  # 原型为cvxopt.matrix(array,dims)，等价于A = matrix([[1.0],[1.0]]）
        # print(A)
        b = matrix([0.1, 1.0])
        result = solvers.qp(P=P, q=q, A=A, b=b)
        print(result)
        print(result['x'])
        '''
        if method == "Markowitz":
            print("\033[0;36;m 开始计算组合权重，采用策略：\033[0m \033[0;34;m Markowitz投资组合 \033[0m")
            # print("\033[0;36;m 开始求解二次规划：\033[0m")
            annual_yield_vector = ex_numpy_vector * total_days_every_year
            w = cp.Variable(n)
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(w, cov_matrix_numpy)),
                              [annual_yield_vector.T @ w == EXPECTED_RETURN,
                               one_matrix @ w == 1])
            prob.solve()

            # print("\nThe optimal value is", prob.value)
            #         # print("A solution w is")
            #         # print(w.value)
            print("\033[0;36;m 完成Markowitz投资组合最优权重二次规划求解，方差最优值为：\033[0m \033[0;34;m {} \033[0m".format(prob.value))

            return w.value

        r_p_list = []
        sigma_p_list = []
        sharpe_ratio_list = []
        weight_list = []
        if method == "MontoCarlo":
            print("\033[0;36;m 开始计算组合权重，采用策略：\033[0m \033[0;34;m Monto Carlo 求解最大夏普比率 \033[0m")
            # 正态分布均值设置为 1 / 50 更符合
            np.random.seed(1)
            risk_free_rate_day = RISK_FREE_RATE / total_days_every_year
            bar = tqdm(list(range(int(MONTO_CARLO_TIMES))),
                       bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))
            for _ in bar:
                # bar.set_description(f"现在到Monto Carlo第{_}次")
                weights = np.random.normal(1 / n, 1.0, n - 1)
                weights_last = 1 - np.sum(weights)
                weights = np.append(weights, weights_last)
                weights_row_vector = np.expand_dims(weights, axis=0)
                yield_avg_vector = np.expand_dims(ex_numpy_vector, axis=0)
                sigma_p = np.sqrt(np.matmul(np.matmul(weights_row_vector, cov_matrix_numpy), weights_row_vector.T))[0][
                    0]
                r_p = np.matmul(weights_row_vector, yield_avg_vector.T)[0][0]

                sharpe_ratio = (r_p - risk_free_rate_day) / sigma_p

                r_p_list.append(r_p)
                sigma_p_list.append(sigma_p)
                sharpe_ratio_list.append(sharpe_ratio)
                weight_list.append(weights)

            r_p_list_numpy = np.array(r_p_list)
            sigma_p_list_numpy = np.array(sigma_p_list)
            sharpe_ratio_list_numpy = np.array(sharpe_ratio_list)
            weight_list_numpy = np.array(weight_list)

            # 最大夏普比率
            max_sharpe_ratio = np.max(sharpe_ratio_list_numpy)
            max_sharpe_ratio_index = np.argmax(sharpe_ratio_list_numpy)

            # 对应的标准差和均值
            sigma_rp = [sigma_p_list_numpy[max_sharpe_ratio_index], r_p_list_numpy[max_sharpe_ratio_index]]

            # r_p与无风险利率组合达到10%收益目标，alpha为投资于无风险利率权重，但其实alpha要接近97%，因为此时市场组合夏普比率最大，日收益率在10%以上，与年收益10%的目标收益和年利率3%无风险利率相去甚远
            alpha = (EXPECTED_RETURN / total_days_every_year - sigma_rp[1]) / (risk_free_rate_day - sigma_rp[1])
            weight_list_numpy_opt_alpha = np.append(weight_list_numpy[max_sharpe_ratio_index], alpha)

            print("\033[0;36;m 完成 Monto Carlo 策略权重求解 \033[0m")
            # 作图
            filename = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(filename):
                os.makedirs(filename)
            plt.figure(figsize=(8, 6))
            plt.style.use('seaborn-dark')
            plt.rcParams['savefig.dpi'] = 300  # 图片像素
            plt.rcParams['figure.dpi'] = 300  # 分辨率
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

            plt.scatter(sigma_p_list_numpy, r_p_list_numpy, c=r_p_list_numpy / sigma_p_list_numpy,
                        marker='o', cmap='coolwarm')
            plt.plot([0, sigma_rp[0]], [risk_free_rate_day, sigma_rp[1]], 'r')
            # plt.annotate('max Sharpe ratio:'.format(max_sharpe_ratio), xy=rp_sigma, xytext=(3, 1.5),
            #              arrowprops=dict(facecolor='black', shrink=0.05),
            #              )
            plt.annotate('max Sharpe ratio:{}'.format(max_sharpe_ratio), xy=sigma_rp)
            plt.xlabel('日标准差')
            plt.ylabel('日收益率')
            plt.colorbar(label='Sharpe ratio')
            plt.title("Monta Carlo抽样{}次获得CAL和有效前沿".format(MONTO_CARLO_TIMES))
            plt.savefig("./images/Montacarlo_CAL_{}_{}_{}".format(MONTO_CARLO_TIMES, starttime, endtime), dpi=300)
            print("\033[0;36;m 完成资本市场线作图 \033[0m")
            return weight_list_numpy_opt_alpha

    @staticmethod
    def get_six_month_map(x_matrix):
        dfx = pd.DataFrame(x_matrix.index, columns=['time'])
        dfx["year"] = pd.to_datetime(pd.DataFrame(x_matrix.index, columns=['time'])['time'], format='%Y-%m-%d').dt.year
        dfx["month"] = pd.to_datetime(pd.DataFrame(x_matrix.index, columns=['time'])['time'],
                                      format='%Y-%m-%d').dt.month

        dfx['yearmonth'] = dfx.apply(lambda r: r['time'][:-2], axis=1)
        dfx = dfx.drop_duplicates(['yearmonth'])

        index_six_month = dfx[(dfx['month'] == 1) | (dfx['month'] == 7)].index.tolist()
        index_slice = int(len(index_six_month) / 2)

        compare_list1 = index_six_month[index_slice:]
        compare_list2 = compare_list1[1:]
        compare_list2.append(x_matrix.shape[0])
        compare_list = list(zip(compare_list1, compare_list2))

        six_map = {k: v for k, v in zip(index_six_month[index_slice:], index_six_month[:index_slice])}
        return six_map, compare_list

    def save_weights_markowitz(self):
        x_matrix_total = self.process_data_x_matrix(DATAPATH)
        six_map, compare_list = self.get_six_month_map(x_matrix_total)

        weight_list = []
        bar = tqdm(six_map.items(), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))
        for k, v in bar:
            start_time = x_matrix_total.iloc[v:k, :].index[0]
            end_time = x_matrix_total.iloc[v:k, :].index[-1]
            bar.set_description(f"进入{start_time}--{end_time}权重计算")
            df_weight = x_matrix_total.iloc[v:k, :]
            total_days = k - v
            weight = self.compute_weight(df_weight, total_days)
            weight_list.append(weight)
        # 保存权重
        filename = os.path.join(os.getcwd(), 'weights')
        if not os.path.exists(filename):
            os.makedirs(filename)
        with open('./weights/weights_Markowitz.pickle', 'wb') as f:
            pickle.dump(weight_list, f)
        with open('./weights/weights_Markowitz.txt', 'w') as f2:
            f2.write(str(weight_list))
        print("\033[0;36;m 权重保存完毕 \033[0m")

    def save_weights_montocarlo(self):
        x_matrix_total = self.process_data_x_matrix(DATAPATH)
        six_map, compare_list = self.get_six_month_map(x_matrix_total)

        weight_list = []
        bar = tqdm(six_map.items(), bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))
        for k, v in bar:
            df_weight = x_matrix_total.iloc[v:k, :]
            total_days = k - v
            start_time = x_matrix_total.iloc[v:k, :].index[0]
            end_time = x_matrix_total.iloc[v:k, :].index[-1]
            bar.set_description(f"进入{start_time}--{end_time}权重计算")

            weight = self.compute_weight(df_weight, total_days, method="MontoCarlo", starttime=start_time, endtime=end_time)
            weight_list.append(weight)
        # 保存权重
        filename = os.path.join(os.getcwd(), 'weights')
        if not os.path.exists(filename):
            os.makedirs(filename)
        with open('./weights/weights_MontoCarlo.pickle', 'wb') as f:
            pickle.dump(weight_list, f)
        with open('./weights/weights_MontoCarlo.txt', 'w') as f2:
            f2.write(str(weight_list))
        print("\033[0;36;m 权重保存完毕 \033[0m")

    def compare_performance(self, method="Markowitz"):
        print("\033[0;36;m 开始与HS300表现比较，比较策略为 \033[0m \033[0;34;m {} \033[0m".format(method))
        total_compare_matrix = pd.DataFrame(columns=['HS300', 'Portfolio', "Period"])
        x_matrix_total_hs300 = self.process_data_contain_hs300(DATAPATH)
        six_map, compare_list = self.get_six_month_map(x_matrix_total_hs300)

        if method == "MontoCarlo_alpha0":
            with open('./weights/weights_{}.pickle'.format("MontoCarlo"), 'rb') as f:
                weight_list = pickle.load(f)
        else:
            with open('./weights/weights_{}.pickle'.format(method), 'rb') as f:
                weight_list = pickle.load(f)

        alpha = 0

        for index, period in enumerate(compare_list):
            if method == "Markowitz":
                weights = weight_list[index]

            elif method == "MontoCarlo":
                weights = weight_list[index][:-1]
                alpha = weight_list[index][-1]

            elif method == "MontoCarlo_alpha0":
                weights = weight_list[index][:-1]

            if period[1] != x_matrix_total_hs300.shape[0]:
                period_day_yield_matrix = x_matrix_total_hs300.iloc[period[0]:period[1] + 1, :]
            else:
                period_day_yield_matrix = x_matrix_total_hs300.iloc[period[0]:period[1], :]
            day_yield_compare_matrix = self.day_yield_compute(period_day_yield_matrix)
            start_time = day_yield_compare_matrix.index[0]
            end_time = day_yield_compare_matrix.index[-1]

            weighted_day_yield = (1 - alpha) * (
                np.matmul(day_yield_compare_matrix.iloc[:, 1:].to_numpy(), weights)) + alpha * RISK_FREE_RATE / 242

            day_yield_compare_matrix['Portfolio'] = pd.DataFrame(weighted_day_yield,
                                                                 index=day_yield_compare_matrix.index)
            day_yield_compare_matrix.rename(columns={0: 'HS300'}, inplace=True)
            period_series = pd.to_datetime(
                pd.DataFrame(day_yield_compare_matrix.index, columns=['time'])['time'], format='%Y-%m-%d')
            dict_data = {'time': period_series.values}

            day_yield_compare_matrix["Period"] = pd.DataFrame(dict_data, index=day_yield_compare_matrix.index)

            # 作图和记录平均收益比较
            hs300_mean = np.mean(day_yield_compare_matrix['HS300'].to_numpy())
            portfolio_mean = np.mean(day_yield_compare_matrix['Portfolio'].to_numpy())
            if hs300_mean < portfolio_mean:
                win = 'Portfolio win!!!'
            else:
                win = 'HS300 win!!!'
            filename = os.path.join(os.getcwd(), 'compare')
            if not os.path.exists(filename):
                os.makedirs(filename)

            with open('./compare/compare_{}.txt'.format(method), 'a') as f:
                f.write(str(start_time) + "--" + str(end_time) + "  " + 'HS300: ' + str(
                    hs300_mean) + "--" + 'Portfolio: ' + str(portfolio_mean) + "---" + win + '\n')

            print("\033[0;36;m 完成\033[0m \033[0;34;m{}--{}\033[0m  \033[0;36;m时间段比较，开始做图 \033[0m".format(start_time,
                                                                                                         end_time))
            self.plot_performance_compare(day_yield_compare_matrix, start_time, end_time, method)
            if index == 0:
                total_compare_matrix = day_yield_compare_matrix
            else:
                total_compare_matrix = pd.concat([total_compare_matrix, day_yield_compare_matrix])

        return total_compare_matrix

    @staticmethod
    def plot_performance_compare(yield_matrix, start_time, end_time, method):
        hs300 = yield_matrix['HS300'].to_numpy()
        portfolio = yield_matrix['Portfolio'].to_numpy()
        period = yield_matrix["Period"].to_numpy()
        # plt.figure(figsize=(15, 9))
        plt.style.use('seaborn-dark')

        fig, ax = plt.subplots()

        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        ax.plot(period, hs300, label='hs300')
        ax.plot(period, portfolio, label='portfolio')

        ax.set(xlabel='日期', ylabel='日收益率',
               title="HS300与{}投资组合收益比较：{}--{}".format(method, start_time, end_time))
        ax.grid()
        ax.legend()

        filename = os.path.join(os.getcwd(), 'images')
        if not os.path.exists(filename):
            os.makedirs(filename)
        plt.savefig("./images/HS300与{}投资组合收益比较：{}--{}".format(method, start_time, end_time), dpi=300)
        plt.close()
        # plt.show()


if __name__ == '__main__':
    matplotlib.use('Agg')

    invent_strate = InvestmentStrategy()

    # # 获取每六个月投资组合权重
    invent_strate.save_weights_markowitz()
    invent_strate.save_weights_montocarlo()

    # # 获取每六个月投资组合收益，并作图
    # method = "Markowitz"
    # method = "MontoCarlo"
    # method = "MontoCarlo_alpha0"
    for method in ["Markowitz", "MontoCarlo", "MontoCarlo_alpha0"]:
        total_compare_yield_matrix = invent_strate.compare_performance(method=method)
        hs300_mean_total = np.mean(total_compare_yield_matrix['HS300'].to_numpy())
        portfolio_mean_total = np.mean(total_compare_yield_matrix['Portfolio'].to_numpy())
        if hs300_mean_total < portfolio_mean_total:
            win = 'Portfolio win!!!'
        else:
            win = 'HS300 win!!!'

        with open('./compare/compare_{}.txt'.format(method), 'a') as f:
            f.write("全部平均：" + str(20150105) + "--" + str(20191230) + "  " + 'HS300: ' + str(
                hs300_mean_total) + "--" + 'Portfolio: ' + str(portfolio_mean_total) + "---" + win + '\n')

        # 做一个总的投资组合和沪深300比较图
        invent_strate.plot_performance_compare(total_compare_yield_matrix, 20150105, 20191230, method)
