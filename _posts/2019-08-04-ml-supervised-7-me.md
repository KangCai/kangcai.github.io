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

最大熵模型（Maximum entropy model）是由最大熵原理推导实现的，掌握最大熵模型只需要弄清楚两个方面的问题：

1. 什么是最大熵原理
2. 最大熵模型是如何运用最大熵原理的

**1.1 什么是最大熵原理**

解释第一个问题，即最大熵原理是什么，需要从熵说起。熵，是热力学中表征物质状态的一种参量，物理意义是体系混乱程度的度量；通信之父香农提出“信息熵”的概念，熵越大的系统，把它搞清楚所需要的信息量也就越大，所以系统是混乱，信息熵就越大；在概率论中，熵越大，随机变量的不确定性越大。假设离散随机变量 X 的概率分布是 P(X)，X 的取值集合是 A，则其熵是

<center>
<img src="https://latex.codecogs.com/gif.latex?H(P)=-\sum_{x}^{x\in&space;A}P(x)log(x)"  />
</center>

且熵满足以下不等式

<center>
<img src="https://latex.codecogs.com/gif.latex?0\leqslant&space;H(P)\leqslant&space;log&space;\left&space;|&space;A&space;\right&space;|"/>
</center>

其中 \|A\| 是 X 的取值个数，当且仅当 X 的分布是均匀分布时右边的等号成立，也就是说，当 X 服从均匀分布时，熵最大。

最大熵原理就是一种运用熵来选择随机变量统计特性最符合客观情况的准则，随机量的概率分布是很难测定的，符合测得这些值的分布可有多种、以至无穷多种，通常，其中有一种分布的熵最大，选用这种具有最大熵的分布作为该随机变量的分布是一种有效的处理方法和准则，可以认为是最符合客观情况的一种选择。举个例子，假设随机变量 X 有 5 个取值 { A, B, C, D, E }，要估计 X 取各个值的概率。满足概率总和为 1 这个约束条件的概率分布有无穷多个，运用最大熵原理，结果就是选择熵最大的那个，P(A)=P(B)=P(C)=P(D)=P(E)=0.2。

在投资时常常讲**不要把所有的鸡蛋放在一个篮子里，这样可以降低风险。在信息处理中，这个原理同样适用。在数学上，这个原理称为最大熵原理**。


**1.2 最大熵模型是如何运用最大熵原理的**

接下来解释第二个问题，即最大熵模型是如何运用最大熵原理的。最大熵模型的建模思想是，学习概率模型时，**在满足已知约束条件的所有可能的模型中，熵最大的模型是最好的模型**。从上面这句话中，可以看到最大熵模型由两个条件组成 —— **满足约束条件**、**熵最大**，最终**学到的是条件概率的表达式**。

首先，满足约束条件指的是什么，还是复用之前的例子，假设随机变量 X 有 5 个取值 { A, B, C, D, E }，要估计 X 取各个值的概率，结果是 P(A)=P(B)=P(C)=P(D)=P(E)=0.2。对于这个问题，再加上一个约束，规定 P(A)+P(B)=0.3，这个就是约束条件，那么模型就需要满足一个约束条件，当然还可以再加一个约束，那么模型就需要满足两个约束条件。

那么，如何用数学公式来表示满足所有约束条件的候选模型呢？

给定一个约束条件，这个约束条件就称之为特征函数（feature function）f(x,y)，它描述了输入 x 和输出 y 之间的某一个事实，表达成数学式是

<center>
<img src="https://latex.codecogs.com/gif.latex?f(x,y)=\left\{\begin{matrix}&space;1,&\text{x&space;and&space;y&space;satisfy&space;some&space;fact}\\&space;0,&\text{otherwise}&space;\end{matrix}\right."/>
</center>

为了保证已知条件成立，我们要求**条件概率P(y|x) 与特征函数f(x,y) 满足以下等式**，

<center>
<img src="https://latex.codecogs.com/gif.latex?\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;f(x,&space;y)=\sum_{x,&space;y}&space;\tilde{P}(x,&space;y)&space;f(x,&space;y)" />
</center>

这个式子左边是特征函数 f(x,y) 关于 P(y|x) 与经验分布 P~(x)的期望值，右边是特征函数 f(x,y) 关于经验分布 P~(x,y)的期望值，使两边相等就使得条件概率满足约束条件；如果有多个约束条件，那么上面的等式就需要列多个。

总之，通过上述等式，我们过滤出了满足了所有约束条件的候选模型，剩下的就是尽可能地是熵最大化，**每一个x，都对应一个特定的概率分布P(y|x)，每一个概率分布都会有一个熵，最大熵就是让所有的 样本概率分布的熵 之和最大**。如下所示

<center>
<img src="https://latex.codecogs.com/gif.latex?H(P)=-\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;\log&space;P(y&space;|&space;x)"  />
</center>

到这里为止，总算可以是从无数种可能的模型中挑选出唯一模型，即最大熵模型。对前面的内容进行整理，我们得出如下的优化数学问题

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&\&space;\min&space;_{R&space;\in&space;\mathrm{C}}-H(P)=\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;\log&space;P(y&space;|&space;x)&space;\\&space;&\begin{array}{ll}{\text&space;{&space;s.t.&space;}}&space;&&space;{E_{P}\left(f_{i}\right)-E_{\tilde{p}}\left(f_{i}\right)=0,&space;\quad&space;i=1,2,&space;\cdots,&space;n}&space;\\&space;{}&space;&&space;{\sum_{y}&space;P(y&space;|&space;x)=1}\end{array}&space;\end{aligned}"/>
</center>

可以看到，s.t.后面的两个等式就是两个约束条件，所以这又是一个典型的带约束的最优化问题，如果对之前介绍的 SVM 算法推导有印象的话，就会意识到可以又通过引入拉格朗日乘子，将问题转化为无约束的最优化问题，具体求解过程 1.3 小节将介绍。

**1.3 求解最大熵模型**

通过引入拉格朗日乘子 w0,w1,...,wn，定义拉格朗日函数 L(P,w)如下，

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(P,&space;w)&space;&&space;\equiv-H(P)&plus;w_{0}\left(1-\sum_{y}&space;P(y&space;|&space;x)\right)&plus;\sum_{i=1}^{n}&space;w_{i}\left(E_{F}\left(f_{i}\right)-E_{P}\left(f_{i}\right)\right)&space;\\&space;&=\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;\log&space;P(y&space;|&space;x)&plus;w_{0}\left(1-\sum_{y}&space;P(y&space;|&space;x)\right)&space;\\&space;&&plus;\sum_{i=1}^{n}&space;w_{i}\left(\sum_{x,&space;y}&space;\tilde{P}(x,&space;y)&space;f_{i}(x,&space;y)-\sum_{x,&space;y}&space;\tilde{P}(x)&space;P(y&space;|&space;x)&space;f_{i}(x,&space;y)\right)&space;\end{aligned}"  />
</center>

求解最大熵模型就转化成了以下问题，

<center>
<img src="https://latex.codecogs.com/gif.latex?\min&space;_{P&space;\in&space;C}&space;\max&space;_{w}&space;L(P,&space;w)" />
</center>

其中 C 为模型的集合，这就是最大熵模型的拉格朗日原始问题，可以进一步转化为拉格朗日对偶问题，

<center>
<img src="https://latex.codecogs.com/gif.latex?\max&space;_{w}&space;\min&space;_{P&space;\in&space;C}&space;L(P,&space;w)" />
</center>

等到了模型，接下来就是求解模型的参数。XXX，todo，参数是什么。。求得 xxx 偏导数，

令偏导数为 0 得到等式

到此为止，除 x 外不含任何其它变量，通过求解上述等式方程依次求出第 i 个迭代量 x。然后将原始参数 w(t) 加上该迭代量 x 就是一次迭代更新后的新参数 w(t+1)。

最大熵模型为，

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

```buildoutcfg
// 计算特征的经验期望值, E_p~(f) = Sum( P~(x,y) * f(x,y) )
for X,y in zip(X_train, Y_train):
    X = tuple(X)
    for idx, val in enumerate(X):
        key = (idx, val, y)
        self.e_feat[key] += self.pxy[(X, y)]
```

3. 初始化参数 w 的每个维度 w_i 为任意值，一般可以设置为0，即

<center>
<img src="https://latex.codecogs.com/gif.latex?w_i^{(0)}&space;=&space;0,&space;i&space;\in&space;\{1,2,3,...,n\}" />
</center>

重复下面的权值更新直至收敛

<center>
<img src="https://latex.codecogs.com/gif.latex?w_i^{(t&plus;1)}&space;=&space;w_i^{(t)}&space;&plus;&space;\frac{1}{C}&space;\log&space;\frac{E_{\hat&space;p}(f_i)}{E_{p^{(n)}}(f_i)},i&space;\in&space;\{1,2,...,n\}"  />
</center>

```buildoutcfg
//迭代找到最优参数
for i in range(self.n_iter):
    delta = self._GIS(X_train, Y_train)
    if np.max(np.abs(delta)) < self.epsilon:
        break
    self.w += delta
    
// GIS算法
def _GIS(self, X_train, Y_train):
    n_feat = len(self.feat_list)
    # 基于当前模型，获取每个特征估计期望, E_p(f) = Sum( P~(x) * P(y|x) * f(x,y) )
    delta = np.zeros(n_feat)
    estimate_feat = defaultdict(float)
    for X,y in zip(X_train, Y_train):
        X = tuple(X)
        py_x = self._cal_py_X(X)[y]
        for idx, val in enumerate(X):
            key = (idx, val, y)
            estimate_feat[key] += self.px[X] * py_x
    # 更新 delta
    for j in range(n_feat):
        feat_key = self.feat_list[j]
        e_feat_exp = self.e_feat[feat_key]
        e_feat_estimate = estimate_feat[feat_key]
        if e_feat_estimate == 0 or e_feat_exp / e_feat_estimate <= 0:
            continue
        delta[j] = 1.0 / self.M * math.log(e_feat_exp / e_feat_estimate)
    delta /= np.sum(delta)
    return delta
```


### 三、表现效果



### 四、与逻辑回归模型的联系

常见的广义线性模型（Generalized Linear Models）有：probit模型、poisson模型、对数线性模型（Log-linear Model）等等。逻辑回归模型（Logistic Regression Model）与 最大熵模型（Maximum Entropy Model）两者同属于**广义线性模型中的对数线性模型**。

两者模型学习一般都采用极大似然估计，优化问题可以形式化为无约束的最优化问题，所以两种模型的最优化算法是通用的，包括：改进的迭代尺度法（IIS）、梯度下降法、拟牛顿法等。

**参考文献**

1. [《统计学习方法》 李航](https://book.douban.com/subject/10590856/)
2. [jianshu: 机器学习面试之最大熵模型](https://www.jianshu.com/p/e7c13002440d)
3. [csdn: 最大熵与逻辑回归的等价性](https://blog.csdn.net/buring_/article/details/43342341)