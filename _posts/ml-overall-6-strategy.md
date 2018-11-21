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

在模型部分，机器学习的学习目标是获得假设空间（模型）的一个最优解，那么接下来如何评判解是优还是不优？**策略部分就是评判“最优模型”（最优参数的模型）的准则和方法**。图1能很好地表示策略是如何做到评判最优的，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/object function.jpg"/>
</center>

如图1所示的函数即经验结构风险的度量函数，也被称为**目标函数（Object function）**，我们的优化目标就是选取最优的模型参数使函数值最小。该函数由两部分组成：经验风险和结构风险，经验风险由代价函数表征，结构风险由正则化函数表征，将经验风险和结构风险放在一起就要考虑两者权衡的问题了。

机器学习中的Error（误差）来源主要为Bias（偏差）、Variance（方差）以及Noise（噪声），由如下公式表示，

<img src="https://latex.codecogs.com/gif.latex?Err(x)=\underbrace{\left&space;[f(x)-\frac{1}{k}\sum_{i=1}^kf(x_i)&space;\right&space;]^2}_{\mathbf{Bias^2}}&space;&plus;&space;\underbrace{&space;\frac{\delta^2}{k}&space;}_{\mathbf{Var}}&plus;&space;\underbrace{\delta^2}_{\mathbf{Noise}}" />

其中Noise无法避免，如果对公式的具体推导感兴趣可以参考[《理解 Bias 与 Variance 之间的权衡》][3]一文。

经验风险最小化就是最小化Bias，结构风险话就是最小化Variance。在统计与机器学习领域权衡 Bias 与 Variance 是一项重要的任务，因为他可以使得用有限训练数据训练得到的模型更好的范化到更多的数据集上。

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/bias and variance 2.jpg"/>
</center>

由于两者不在同一个度量衡上，所以通常会在正则化函数前面加一个大于0的正则化系数 λ ：λ 过小意味着正则化约束弱，对复杂模型的惩罚小，会导致过拟合。也有办法自动选择正则化参数 λ 。

代价函数是样本集的损失函数之和，本文的第二节直接主要介绍损失函数；正则化是只对wx+b中参数w的约束，通常按照惯例是不管b的，但如果在实际应用时在正则化函数里将b也加上的话，影响不大，基本没差别，本文的第三节介绍正则化。

### 二、损失函数

损失函数，即Loss Function，

### 三、正则化

正则化，即Regularization

3. [cnblogs: 理解 Bias 与 Variance 之间的权衡][3]

[3]: (https://www.cnblogs.com/ooon/p/5711516.html)