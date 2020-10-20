import math
import os
import numpy as np
from config import K, M_STEP, VOLATILITY, EXPIRES_ANNUALIZE, S_0, RISK_FREE_RATE_OPTION
from draw_tree_picture import create_btree_by_list
from scipy.stats import norm


class OptionPrice:
    def __init__(self, S, K, r, sigma, t, q=0, steps=2):
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.steps = steps
        self.q = q
        self.u = np.exp(sigma * np.sqrt(t / steps))
        self.d = 1 / self.u
        self.P = (np.exp((r - q) * t / steps) - self.d) / (self.u - self.d)
        self.prices = np.zeros(steps + 1)
        self.c_values = np.zeros(steps + 1)
        # self.continuous_compound_interest = np.log(1 + r)

    __doc__ = '''S:标的资产初始价格；
        K:期权的执行价格;
        r:年化无风险利率；
        q:连续分红的红利率；
        sigma:标的资产连续复利收益率的标准差；
        t:以年表示的时间长度；
        steps:二叉树的步长。'''

    def binarytree_american_put(self):
        list_tree = []

        self.prices[0] = self.S * self.d ** self.steps
        self.c_values[0] = np.maximum(K - self.prices[0], 0)
        for i in range(1, self.steps + 1):
            self.prices[i] = self.prices[i - 1] * (self.u ** 2)
            self.c_values[i] = np.maximum(K - self.prices[i], 0)

        # 作图保留两位小数
        list_tree.append(
            list(zip([float('{:.2f}'.format(i)) for i in self.prices],
                     [float('{:.2f}'.format(i)) for i in self.c_values])))

        for j in range(self.steps, 0, -1):
            for i in range(0, j):
                self.prices[i] = self.prices[i + 1] * self.d
                self.c_values[i] = np.maximum(
                    (self.P * self.c_values[i + 1] + (1 - self.P) * self.c_values[i]) * np.exp(
                        -self.r * self.t / self.steps),
                    K - self.prices[i])

            # 作图保留两位小数
            list_tree.append(list(
                zip([float('{:.2f}'.format(p)) for p in self.prices[:j]],
                    [float('{:.2f}'.format(p)) for p in self.c_values[:j]])))

        list_tree.reverse()
        list_tree_bt = []
        for i in list_tree:
            if len(i) > 2:
                list_tree_bt.append(i[0])
                for j in range(1, len(i) - 1):
                    list_tree_bt.append(i[j])
                    list_tree_bt.append(i[j])
                list_tree_bt.append(i[-1])
            else:
                for j in range(len(i)):
                    list_tree_bt.append(i[j])

        return self.c_values[0], list_tree, list_tree_bt

    def binarytree_european_put(self):
        list_tree = []

        self.prices[0] = self.S * self.d ** self.steps
        self.c_values[0] = np.maximum(K - self.prices[0], 0)
        for i in range(1, self.steps + 1):
            self.prices[i] = self.prices[i - 1] * (self.u ** 2)
            self.c_values[i] = np.maximum(K - self.prices[i], 0)

        # 作图保留两位小数
        list_tree.append(
            list(zip([float('{:.2f}'.format(i)) for i in self.prices],
                     [float('{:.2f}'.format(i)) for i in self.c_values])))

        for j in range(self.steps, 0, -1):
            for i in range(0, j):
                self.prices[i] = self.prices[i + 1] * self.d
                self.c_values[i] = (self.P * self.c_values[i + 1] + (1 - self.P) * self.c_values[i]) * np.exp(
                    -self.r * self.t / self.steps)

                # 作图保留两位小数
                list_tree.append(list(
                    zip([float('{:.2f}'.format(p)) for p in self.prices[:j]],
                        [float('{:.2f}'.format(p)) for p in self.c_values[:j]])))

        list_tree.reverse()
        list_tree_bt = []
        for i in list_tree:
            if len(i) > 2:
                list_tree_bt.append(i[0])
                for j in range(1, len(i) - 1):
                    list_tree_bt.append(i[j])
                    list_tree_bt.append(i[j])
                list_tree_bt.append(i[-1])
            else:
                for j in range(len(i)):
                    list_tree_bt.append(i[j])
        return self.c_values[0], list_tree, list_tree_bt

    def binarytree_european_call(self):
        list_tree = []

        self.prices[0] = self.S * self.d ** self.steps
        self.c_values[0] = np.maximum(self.prices[0] - K, 0)
        for i in range(1, self.steps + 1):
            self.prices[i] = self.prices[i - 1] * (self.u ** 2)
            self.c_values[i] = np.maximum(self.prices[i] - K, 0)

        # 作图保留两位小数
        list_tree.append(
            list(zip([float('{:.2f}'.format(i)) for i in self.prices],
                     [float('{:.2f}'.format(i)) for i in self.c_values])))

        for j in range(self.steps, 0, -1):
            for i in range(0, j):
                self.c_values[i] = (self.P * self.c_values[i + 1] + (1 - self.P) * self.c_values[i]) * np.exp(
                    -self.r * self.t / self.steps)
                self.prices[i] = self.prices[i + 1] * self.d

            # 作图保留两位小数
            list_tree.append(list(
                zip([float('{:.2f}'.format(p)) for p in self.prices[:j]],
                    [float('{:.2f}'.format(p)) for p in self.c_values[:j]])))

        list_tree.reverse()
        list_tree_bt = []
        for i in list_tree:
            if len(i) > 2:
                list_tree_bt.append(i[0])
                for j in range(1, len(i) - 1):
                    list_tree_bt.append(i[j])
                    list_tree_bt.append(i[j])
                list_tree_bt.append(i[-1])
            else:
                for j in range(len(i)):
                    list_tree_bt.append(i[j])

        return self.c_values[0], list_tree, list_tree_bt

    def black_scholes_option(self, option='call'):
        """
        S: spot price
        K: strike price
        t: time to maturity
        r: risk-free interest rate
        sigma: standard deviation of price of underlying asset
        """
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))

        if option == 'call':
            p = (self.S * norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.t) * norm.cdf(d2, 0.0, 1.0))
        elif option == 'put':
            p = (self.K * np.exp(-self.r * self.t) * norm.cdf(-d2, 0.0, 1.0) - self.S * norm.cdf(-d1, 0.0, 1.0))
        else:
            return None
        return p

    @staticmethod
    def draw_binarytree(list_tree, picturepath):
        tree = create_btree_by_list(list_tree)
        tree.print_tree(save_path=picturepath, label=False)


if __name__ == '__main__':
    filename = os.path.join(os.getcwd(), 'option_result')
    if not os.path.exists(filename):
        os.makedirs(filename)
    # 第二小问美式看跌
    op = OptionPrice(S=S_0, K=K, r=RISK_FREE_RATE_OPTION, q=0, sigma=VOLATILITY, t=EXPIRES_ANNUALIZE, steps=M_STEP)
    binarytree_american_put_price, resultlist, list_binary_tree_forpic = op.binarytree_american_put()
    if M_STEP <= 10:
        op.draw_binarytree(list_binary_tree_forpic,
                           os.path.join(os.getcwd(), 'option_result/binarytree_american_put_{}_step'.format(M_STEP)))
    with open('./option_result/binarytree_american_put_{}_step.txt'.format(M_STEP), 'w') as f:
        f.write("Option price: " + str(binarytree_american_put_price) + '\n')
        f.write(str(resultlist))

    # 第一小问欧式看跌
    for M_STEP in [10, 50, 100]:
        op = OptionPrice(S=S_0, K=K, r=RISK_FREE_RATE_OPTION, q=0, sigma=VOLATILITY, t=EXPIRES_ANNUALIZE, steps=M_STEP)
        binarytree_european_put_price, resultlist, list_binary_tree_forpic = op.binarytree_european_put()
        if M_STEP <= 10:
            op.draw_binarytree(list_binary_tree_forpic,
                               os.path.join(os.getcwd(),
                                            'option_result/binarytree_european_put_{}_step'.format(M_STEP)))
        with open('./option_result/binarytree_european_put_{}_step.txt'.format(M_STEP), 'w') as f:
            f.write("Option price: " + str(binarytree_european_put_price) + '\n')
            f.write(str(resultlist))

    # 第一小问BS公式看跌
    op = OptionPrice(S=S_0, K=K, r=RISK_FREE_RATE_OPTION, q=0, sigma=VOLATILITY, t=EXPIRES_ANNUALIZE, steps=M_STEP)
    p = op.black_scholes_option("put")
    with open('./option_result/black_scholes_put.txt', 'w') as f:
        f.write("Option price: " + str(p) + '\n')
