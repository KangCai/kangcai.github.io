---
layout: post
title: "机器学习·总览篇 VI"
subtitle: "三要素之策略"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)

### 一、三要素之策略

在模型部分，机器学习的学习目标是获得假设空间（模型）的一个最优解，那么接下来如何评判解是优还是不优？比如对于一个回归任务，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/regression.png"/>
</center>

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/classification.png"/>
</center>

**策略部分就是评判“最优模型”（最优参数的模型）的准则和方法**。图1能很好地表示策略是如何做到评判最优的，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/object function.jpg"/>
</center>

如图1所示的函数即经验结构风险的度量函数，也被称为**目标函数（Object function）**，我们的优化目标就是选取最优的模型参数使函数值最小。该函数由两部分组成：经验风险和结构风险，经验风险由代价函数表征，结构风险由正则化函数表征，将经验风险和结构风险放在一起就要考虑权衡问题了，如何权衡两者才能尽可能地减少学习误差。机器学习中的Error（误差）来源主要为Bias（偏差）、Variance（方差）以及Irreducible error（Noise，噪声），由如下公式表示，

<center>
<img src="https://latex.codecogs.com/gif.latex?Err(x)&space;=&space;\left(E[\hat{f}(x)]-f(x)\right)^2&space;&plus;&space;E\left[\left(\hat{f}(x)-E[\hat{f}(x)]\right)^2\right]&space;&plus;\sigma_e^2" />
</center>
<center>
<img src="https://latex.codecogs.com/gif.latex?Err(x)&space;=&space;\mathrm{Bias}^2&space;&plus;&space;\mathrm{Variance}&space;&plus;&space;\mathrm{Irreducible\&space;Error}" />
</center>

其中Bias是模型真实预测结果的期望与理论误差下界模型预测结果的偏差，Variance是指相同大小、不同训练集训练出来的模型的预测结果与模型预测期望的方差，Irreducible error是理论误差下界模型的泛化误差，它刻画了问题本身的难度。关于Bias和Variance更形象的表示如图2所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/bias and variance 1.png"/>
</center>

其中的一个点表示一个模型的预测效果，低Bias能让模型的预测期望就在靶心，低Variance能让模型预测表现地更稳定。经验风险会直接影响Bias，结构风险会直接影响Variance，在统计与机器学习领域权衡 Bias 与 Variance 是一项重要的任务，因为它可以使得用有限训练数据训练得到的模型更好地范化到更多的数据集上，如图3所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/bias and variance 2.png"/>
</center>

从图3中可以看到，Bias和Viriance虽然不能两全，但通过好的权衡，达到总体来说好地效果，如图3中的Total Error黑色曲线的最低点。同理，对于目标函数来说，代价函数和正则化项本身不在同一个度量衡上，所以通常会在正则化函数前面加一个大于0的正则化系数 λ ：λ 过小意味着正则化约束弱，对模型的复杂度惩罚过小，会导致过拟合和高Virance；λ 过小意味着代价函数约束弱，对模型的复杂度惩罚过大，会导致欠拟合和高Bias。选取合适的 λ 是一门“艺术”，跟下一篇文章确定学习率（learning rate）类似有很多种方法，当然也有办法自动选择正则化参数 λ，具体选取方法可参考[[4]][4]和[[5]][5]两篇文章。

对目标函数的含义有了大概的了解，接下来重点是代价函数和正则化项：
1. 代价函数是样本集的损失函数之和，本文的第二节直接主要介绍**损失函数**；
2. 正则化是对wx+b中参数w的约束，本文的第三节介绍**正则化**。

### 二、损失函数

损失函数，即Loss Function，如上文所说，是对经验误差的惩罚。下面我们将比较全面地介绍各种损失函数，为了方便理解，可以将分类问题和回归问题的损失函数分开考虑。

##### 2.1 分类模型的损失函数

分类模型的损失函数按函数形式分有以下常见类别：

|  | 应用模型 | 函数形式 | 
| :-----------:| :----------: | :----------: |
| 0-1 loss | PLA [[6]][6] |  <img src="https://latex.codecogs.com/gif.latex?L_{01}\left&space;(&space;m&space;\right&space;)=\begin{cases}&space;0&space;&&space;\text{&space;if&space;}&space;m\geqslant&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;m<&space;0&space;\end{cases}" /> |
| Hinge loss| SVM | <img src="https://latex.codecogs.com/gif.latex?max\left&space;(&space;0,1-m&space;\right&space;)" /> |
| log loss | LR |  <img src="https://latex.codecogs.com/gif.latex?log\left&space;(&space;1&plus;exp\left&space;(&space;-m&space;\right&space;)&space;\right&space;)" /> |
| exponential loss | Boosting | <img src="https://latex.codecogs.com/gif.latex?exp\left&space;(&space;-m&space;\right&space;)" /> |
| perceptron loss | Perceptron | <img src="https://latex.codecogs.com/gif.latex?max\left&space;(&space;0,\;&space;-m&space;\right&space;)" />|

0-1 loss也称Goal Standard，非凸函数，一般不会直接用，通常在实际的使用中将0-1损失函数作为一个标准，选择它的代理函数作为损失函数，不同的机器学习模型对应的损失函数通常比较固定，有一一对应的关系，比如：SVM用的是Hinge loss，LR用的是log loss，Adaboost用的是exp loss，感知器用的是perceptron loss。

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/loss function decision.png"/>
</center>

##### 2.2 回归模型的损失函数

|  | 应用模型 | 函数形式 | 
| :-----------:| :----------: | :----------: |
| squared loss | OLS |  <img src="https://latex.codecogs.com/gif.latex?a^2" /> |
| absolute loss |   |  <img src="https://latex.codecogs.com/gif.latex?abs(a)" /> |
| log-cosh loss | XGBoost | <img src="https://latex.codecogs.com/gif.latex?log(cosh(a))"/> |
| Huber loss | | <img src="https://latex.codecogs.com/gif.latex?L_\delta(a)=\left&space;\{&space;\begin{array}{ll}&space;\frac12a^2,&\textrm{if&space;}&space;abs(a)\leq\delta,\\&space;\delta\cdot(abs(a)-\frac12\delta),&\textrm{otherwise.}&space;\end{array}&space;\right."  /> |
| ϵ−insensitive loss | SVR | <img src="https://latex.codecogs.com/gif.latex?L_\varepsilon(a)=\begin{cases}0,&\text{if&space;}abs(a)\leq\varepsilon\text;\\abs(a)-\varepsilon,&\text{otherwise.}\end{cases}" /> |
| Quantile loss |  |<img src="https://latex.codecogs.com/gif.latex?L_\gamma(a)=\begin{cases}(1-\gamma)\cdot&space;abs(a)&space;&\text{if&space;}a<0;&space;\\&space;\gamma&space;\cdot&space;abs(a)&space;&&space;\text{otherwise.}\end{cases}" /> |

表格中R表示Regression，abs(a)表示a的绝对值（由于latex的绝对值符号与markdown格式的表格冲突了，只能用abs来表示一下）。OLS（最小二乘法）用的是squre loss，XGBoost一般用的是log-cosh loss，SVR用的是ϵ−insensitive loss，对于神经网络或者直接回归，很多不同的损失函数互相替换都可以work，只有表现效果上的差别。

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/loss function regression.png"/>
</center>

### 三、正则化

正则化，即Regularization

正则化是对wx+b中参数w的约束，通常按照惯例是不管b的，但如果在实际应用时在正则化函数里将b也加上的话，影响不大，效果基本没差别。


[3][3]

1. [blog: Understanding the Bias-Variance Tradeoff][1]
2. [cnblogs: 理解 Bias 与 Variance 之间的权衡][2]
3. [cnblogs: 机器学习之正则化][3]
4. [DTU: Choosing the Regularization Parameter][4]
5. [Paper: Methods for Choosing the Regularization Parameter][5]
6. [blog: Perceptron Learning Algorithm][6]
7. [csdn: 机器学习中的常见问题——损失函数][7]

[1]: (http://scott.fortmann-roe.com/docs/BiasVariance.html)
[2]: (https://www.cnblogs.com/ooon/p/5711516.html)
[3]: (https://www.cnblogs.com/jianxinzhou/p/4083921.html)
[4]: (http://www2.compute.dtu.dk/~pcha/DIP/chap5.pdf)
[5]: (https://projecteuclid.org/download/pdf_1/euclid.pcma/1416323374)
[6]: (http://kubicode.me/2015/08/06/Machine%20Learning/Perceptron-Learning-Algorithm/)
[7]: (https://blog.csdn.net/google19890102/article/details/50522945)