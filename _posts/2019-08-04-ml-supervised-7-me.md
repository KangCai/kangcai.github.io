---
layout: post
title: "机器学习 · 监督学习篇 VII"
subtitle: "最大熵模型"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

> 最大熵模型（Maximum entropy model）

> 文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

### 一、概念

特征函数（feature function）f(x,y) 描述输入 x 和输出 y 之间的某一个事实，其定义是

<center>
<img src="https://latex.codecogs.com/gif.latex?f(x,y)=\left\{\begin{matrix}&space;1,&\text{x&space;and&space;y&space;satisfy&space;some&space;fact}\\&space;0,&\text{otherwise}&space;\end{matrix}\right."/>
</center>

...

求得 xxx 偏导数，

令偏导数为 0 得到等式

到此为止，除 x 外不含任何其它变量，通过求解上述等式方程依次求出第 i 个迭代量 x。然后将原始参数 w(t) 加上该迭代量 x 就是一次迭代更新后的新参数 w(t+1)。


### 二、算法步骤

目的是求解上述方程，即

XXX

**迭代尺度法（Generalized Iterative Scaling, GIS）**

1. 根据训练集特征 X 的每个维度上的值 x 与 y 组成的 (x, y)对，求经验概率

```buildoutcfg
// 统计 (x,y) 的联合概率分布，x 的经验边缘分布, x(x,y)
self.N, self.M = X_train.shape
feat_set = set()
for X,y in zip(X_train, Y_train):
    X = tuple(X)
    self.px[X] += 1.0 / self.N
    self.pxy[(X, y)] += 1.0 / self.N
    for idx, val in enumerate(X):
        key = (idx, val, y)
        feat_set.add(key)
self.feat_list = list(feat_set)
```

2. 初始化参数 w 的每个维度 w_i 为任意值，一般可以设置为0，即

<center>
<img src="https://latex.codecogs.com/gif.latex?w_i^{(0)}&space;=&space;0,&space;i&space;\in&space;\{1,2,3,...,n\}" />
</center>

重复下面的权值更新直至收敛

<center>
<img src="https://latex.codecogs.com/gif.latex?w_i^{(t&plus;1)}&space;=&space;w_i^{(t)}&space;&plus;&space;\frac{1}{C}&space;\log&space;\frac{E_{\hat&space;p}(f_i)}{E_{p^{(n)}}(f_i)},i&space;\in&space;\{1,2,...,n\}"  />
</center>


改进的迭代尺度法（Improved Iterative Scaling, IIS）

### 三、表现效果

最大熵模型（Maximum Entropy Model）与逻辑回归模型（Logistic Regression Model）两者同属于广义线性模型（Generalized Linear Models），有诸多相似之处，李航老师在《统计学习方法》书中也是把两种模型放在同一章来讲的，具体来讲有以下异同：

1. 同属于广义线性模型，但逻辑回归模型假设条件概率分布是XXX，最大熵模型假设的条件概率分布是YYY。

2. 两者模型学习一般都采用极大似然估计，优化问题可以形式化为无约束的最优化问题，最优化算法有迭代尺度法、改进的迭代尺度法、梯度下降法、拟牛顿法。

**参考文献**

1. [《统计学习方法》 李航](https://book.douban.com/subject/10590856/)
2. [jianshu: 机器学习面试之最大熵模型](https://www.jianshu.com/p/e7c13002440d)