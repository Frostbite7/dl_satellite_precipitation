import numpy as np
# import matplotlib.pyplot as plt
import time

start_time = time.time()  # 计时


# 正态分布密度函数
def phi(x, miu, sigma):
    y = np.exp((-(x - miu) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return y


# 用正态核对序列a进行smooth
def pdf(x, a, h):
    n = len(a)
    y = 0
    for i in range(n):
        y = y + phi(x, a[i], h)
    y = y / n
    return y


# 对a求导，这里a不是序列，是其中一个值
def dpdf(x, a, h):
    dy = np.exp((-(x - a) ** 2) / (2 * h ** 2)) * (x - a) / h ** 2
    dy = dy / (np.sqrt(2 * np.pi) * h)
    return dy


# 计算梯度函数，输出一个序列，对b序列中每个值的偏导
def num_dkl(a, b, h, min, max, n_bins):
    n = len(a)
    div_da = [0] * n
    step = (max - min) / n_bins
    x = step / 2 + min
    for i in range(n_bins):
        temp = step * pdf(x, a, h) / pdf(x, b, h)
        for j in range(n):
            div_da[j] = div_da[j] - temp * dpdf(x, b[j], h) / n
        x = x + step
    return div_da


# 数值积分求kl散度
def num_kl_div(a, b, h, min, max, n_bins):
    div = 0
    step = (max - min) / n_bins
    x = step / 2 + min
    for i in range(n_bins):
        div = div + step * pdf(x, a, h) * np.log(pdf(x, a, h) / pdf(x, b, h))
        # print(pdf(x,a,h),pdf(x,b,h))
        # print(div)
        x = x + step
    return div


# 近似公式求kl散度，用不上
def appr_kl_div(a, b, h):
    n = len(a)
    div = 0
    for i in range(n):
        temp1 = 0
        temp2 = 0
        for j in range(n):
            temp1 = temp1 + np.exp(-((a[i] - a[j]) ** 2) / (2 * h ** 2))
            temp2 = temp2 + np.exp(-((a[i] - b[j]) ** 2) / (2 * h ** 2))
        div = div + np.log(temp1 / temp2)
    div = div / n
    return div


# 计算mean squared error
def com_mse(a, b):
    c = (a - b) * (a - b)
    return np.mean(c)


h = 2.5
# a = np.array([i for i in range(50)] + [1] * 500)
a = np.fromfile('test_data/rain_value_list.bin')
n = len(a)
b = np.array([0] * n)

div_n = num_kl_div(a, b, h, -10, 30, 100)
# div_a = appr_kl_div(a, b, h)
mse = com_mse(a, b)
div_da = num_dkl(a, b, h, -10, 50, 20)
print(div_n, mse)
print(div_da)

# x = np.linspace(5, 70, 1000)
# y = pdf(x, a, h)
# plt.plot(x, y)
# plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
