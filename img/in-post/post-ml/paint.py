# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import math

'''
# paint 2018-10-26-ml-overall-bayes-x.png
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
'''

'''
# paint decision loss function curve
def func1(xl):
    y = []
    for x in xl:
        val = 0 if x >= 0 else 1
        y.append(val)
    return y

def func2(xl):
    y = []
    for x in xl:
        y.append(max(0, 1-x))
    return y

def func3(xl):
    y = []
    for x in xl:
        y.append(math.log2(1+math.e**(-x)))
    return y

def func4(xl):
    y = []
    for x in xl:
        y.append(math.e**(-x))
    return y

def func5(xl):
    y = []
    for x in xl:
        y.append(max(0, -x))
    return y

x_ = np.arange(-20,20,0.001)
y_ = func1(x_)
y1_ = func2(x_)
y2_ = func3(x_)
y3_ = func4(x_)
y4_ = func5(x_)
plt.figure(figsize=(12, 4.5))

plt.subplot(121)
plt.xlabel('y · f(x)')
plt.ylabel('L(y, f(x))')
plt.plot(x_, y_, label='0-1 loss')
plt.plot(x_, y1_, label='Hinge loss')
plt.plot(x_, y2_, label='log loss')
plt.plot(x_, y3_, label='exponential loss')
plt.plot(x_, y4_, label='perceptron loss')
# 设置x坐标轴刻度,
plt.xticks(range(-3, 4))
# 设置y坐标轴刻度及标签, $$是设置字体
#plt.yticks(np.linspace(0, 0.12, 7), np.linspace(0, 12, 7))
# 获取当前的坐标轴, gca = get current axis
ax = plt.gca()
# 设置x坐标轴为下边框
ax.xaxis.set_ticks_position('bottom')
# 设置y坐标轴为左边框
ax.yaxis.set_ticks_position('left')
plt.xlim(-3, 3)
plt.ylim(0, 5)
plt.legend()

plt.subplot(122)
plt.xlabel('y · f(x)')
plt.ylabel('L(y, f(x))')
plt.plot(x_, y_, label='0-1 loss')
plt.plot(x_, y1_, label='Hinge loss')
plt.plot(x_, y2_, label='log loss')
plt.plot(x_, y3_, label='exponential loss')
plt.plot(x_, y4_, label='perceptron loss')
# 设置x坐标轴刻度,
plt.xticks(range(-20, 21, 5))
# 设置y坐标轴刻度及标签, $$是设置字体
#plt.yticks(np.linspace(0, 0.12, 7), np.linspace(0, 12, 7))
# 获取当前的坐标轴, gca = get current axis
ax = plt.gca()
# 设置x坐标轴为下边框
ax.xaxis.set_ticks_position('bottom')
# 设置y坐标轴为左边框
ax.yaxis.set_ticks_position('left')
plt.xlim(-20, 20)
plt.ylim(0, 30)
plt.legend()

plt.show()
'''

'''
# paint regression loss function curve
def func1(xl):
    y = []
    for x in xl:
        y.append(x**2)
    return y

def func2(xl):
    y = []
    for x in xl:
        y.append(math.fabs(x))
    return y

def func3(xl):
    y = []
    for x in xl:
        y.append(math.log2(math.cosh(x)))
    return y

def func4(xl):
    y = []
    for x in xl:
        val = 0.5 * x ** 2 if math.fabs(x) < 5 else 5 * (math.fabs(x) - 2.5)
        y.append(val)
    return y

def func5(xl):
    y = []
    for x in xl:
        val = 0 if math.fabs(x) < 1 else math.fabs(x) - 1
        y.append(val)
    return y

def func6(xl):
    y = []
    for x in xl:
        val = (1 - 0.1) * math.fabs(x) if x < 0 else 0.1 * math.fabs(x)
        y.append(val)
    return y

x_ = np.arange(-4,4,0.001)
y_ = func1(x_)
y1_ = func2(x_)
y2_ = func3(x_)
y3_ = func4(x_)
y4_ = func5(x_)
y5_ = func6(x_)
plt.figure(figsize=(8, 4))

plt.xlabel('y - f(x)')
plt.ylabel('L(y, f(x))')
plt.plot(x_, y_, label='squared loss')
plt.plot(x_, y1_, label='absolute loss')
plt.plot(x_, y2_, label='log-cosh loss')
plt.plot(x_, y3_, label='Huber loss(δ=5)')
plt.plot(x_, y4_, label='ϵ−insensitive loss(ϵ=1)')
plt.plot(x_, y5_, label='Quantile loss(γ=0.1)')
# 设置x坐标轴刻度,
plt.xticks(range(-4, 5))
# 设置y坐标轴刻度及标签, $$是设置字体
#plt.yticks(np.linspace(0, 0.12, 7), np.linspace(0, 12, 7))
# 获取当前的坐标轴, gca = get current axis
ax = plt.gca()
# 设置x坐标轴为下边框
ax.xaxis.set_ticks_position('bottom')
# 设置y坐标轴为左边框
ax.yaxis.set_ticks_position('left')
plt.xlim(-4, 4)
plt.ylim(0, 4)
plt.legend()

plt.show()
'''

def func1(xl):
    y = []
    for x in xl:
        y.append(x**2)
    return y

def func2(xl):
    y = []
    for x in xl:
        y.append(math.fabs(x))
    return y

x_ = np.arange(-1,1,0.001)
y_ = func1(x_)
y1_ = func2(x_)
plt.figure(figsize=(8, 4))

plt.xlabel('w')
plt.ylabel('L')
plt.plot(x_, y1_, label='L1 (Lasso)')
plt.plot(x_, y_, label='L2 (Ridge)')
# 设置x坐标轴刻度,
plt.xticks(np.linspace(-1, 1, 5))
# 设置y坐标轴刻度及标签, $$是设置字体
#plt.yticks(np.linspace(0, 0.12, 7), np.linspace(0, 12, 7))
# 获取当前的坐标轴, gca = get current axis
ax = plt.gca()
# 设置x坐标轴为下边框
ax.xaxis.set_ticks_position('bottom')
# 设置y坐标轴为左边框
ax.yaxis.set_ticks_position('left')
plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.legend()

plt.show()