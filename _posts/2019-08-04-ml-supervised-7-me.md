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


**步骤一 模型定义**

给定一个已知条件，

这个已知条件就称之为特征函数（feature function）f(x,y)，它描述了输入 x 和输出 y 之间的某一个事实，表达成数学式是

<center>
<img src="https://latex.codecogs.com/gif.latex?f(x,y)=\left\{\begin{matrix}&space;1,&\text{x&space;and&space;y&space;satisfy&space;some&space;fact}\\&space;0,&\text{otherwise}&space;\end{matrix}\right."/>
</center>

**为了保证已知条件地成立，我们要求条件概率与特征函数满足以下等式**，

<center>
<img src="https://latex.codecogs.com/gif.latex?\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;f(x,&space;y)=\sum_{x,&space;y}&space;\tilde{P}(x,&space;y)&space;f(x,&space;y)" />
</center>

满足了约束条件，剩下的就是尽可能地是熵最大化，**每一个x，都对应一个特定的概率分布P(y|x)，每一个概率分布都会有一个熵，最大熵就是让所有的 样本概率分布的熵 之和最大**。如下所示

<center>
<img src="https://latex.codecogs.com/gif.latex?H(P)=-\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;\log&space;P(y&space;|&space;x)"  />
</center>

**步骤二 求解模型**

对前面的内容进行整理，我们得出如下的优化数学问题

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&\&space;\min&space;_{R&space;\in&space;\mathrm{C}}-H(P)=\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;\log&space;P(y&space;|&space;x)&space;\\&space;&\begin{array}{ll}{\text&space;{&space;s.t.&space;}}&space;&&space;{E_{P}\left(f_{i}\right)-E_{\tilde{p}}\left(f_{i}\right)=0,&space;\quad&space;i=1,2,&space;\cdots,&space;n}&space;\\&space;{}&space;&&space;{\sum_{y}&space;P(y&space;|&space;x)=1}\end{array}&space;\end{aligned}"/>
</center>

求得 xxx 偏导数，

令偏导数为 0 得到等式

到此为止，除 x 外不含任何其它变量，通过求解上述等式方程依次求出第 i 个迭代量 x。然后将原始参数 w(t) 加上该迭代量 x 就是一次迭代更新后的新参数 w(t+1)。

最大熵模型为

### 二、算法实现

目的是求解上述方程，即

XXX

**迭代尺度法（Generalized Iterative Scaling, GIS）**

1. 统计 (X,y) 的联合概率分布 P(X, y)，X 的经验边缘分布 P(X)

```buildoutcfg
// 统计 (X,y) 的联合概率分布 P(X, y)，X 的经验边缘分布 P(X)
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


### 三、表现效果



### 四、与逻辑回归模型的联系

逻辑回归模型（Logistic Regression Model）与 最大熵模型（Maximum Entropy Model）两者**同属于广义线性模型（Generalized Linear Models）**，逻辑回归模型假设条件概率分布是二项式分布，最大熵模型假设的条件概率分布是多项式分布，所以要更加精确地描述两者关系，应该说**逻辑回归模型也是最大熵模型的一种特殊情况（多项式的 k=2 的情形）**。

由于逻辑回归模型是最大熵模型的一种特殊情况，所以逻辑回归模型最优化算法都能应用到最大熵模型中。两种模型学习一般都采用极大似然估计，最优化算法有改进的迭代尺度法、梯度下降法、拟牛顿法等。

**参考文献**

1. [《统计学习方法》 李航](https://book.douban.com/subject/10590856/)
2. [jianshu: 机器学习面试之最大熵模型](https://www.jianshu.com/p/e7c13002440d)
3. [csdn: 最大熵与逻辑回归的等价性](https://blog.csdn.net/buring_/article/details/43342341)