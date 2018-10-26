# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import math

def func1(xl):
    y = []
    y_sum = 0
    for x in xl:
        val = x * x * math.pow(math.e, -(x - 0.5) *(x - 0.5) / (2 * 0.1 * 0.1)) / (math.sqrt(2 * math.pi) * 0.1)
        y.append(val)
        y_sum += 0.001 * val
    for i in range(len(y)):
        y[i] /= y_sum
    return y

def func2(xl):
    y = []
    for x in xl:
        y.append(math.pow(math.e, -(x - 0.5) * (x - 0.5) / (2 * 0.1 * 0.1)) / (math.sqrt(2 * math.pi) * 0.1))
    return y

def func3(xl):
    y = []
    y_sum = 0
    for x in xl:
        val = math.pow(x, 100) * math.pow(math.e, -(x - 0.5) *(x - 0.5) / (2 * 0.1 * 0.1)) / (math.sqrt(2 * math.pi) * 0.1)
        y.append(val)
        y_sum += 0.001 * val
    for i in range(len(y)):
        y[i] /= y_sum * 10
    return y

plt.figure(figsize=(8, 4))
plt.xlabel('θ')
plt.ylabel('P(θ)')
x_ = np.arange(0,1,0.001)

'''function below'''
y_ = func1(x_)
y1_ = func2(x_)
y2_ = func3(x_)

plt.plot(x_, y1_, label='Prior Distribution')
plt.plot(x_, y_, label='K=2 Posterior Distribution')
plt.plot(x_, y2_, label='K=100 Posterior Distribution')

# 设置x坐标轴刻度,
plt.xticks(np.arange(0,1.000001,0.1))
# 设置y坐标轴刻度及标签, $$是设置字体
#plt.yticks(np.linspace(0, 0.12, 7), np.linspace(0, 12, 7))
# 获取当前的坐标轴, gca = get current axis
ax = plt.gca()
# 设置x坐标轴为下边框
ax.xaxis.set_ticks_position('bottom')
# 设置y坐标轴为左边框
ax.yaxis.set_ticks_position('left')
plt.xlim(0, 1)
plt.ylim(0, 6)
plt.grid(True)
plt.legend()
plt.show()