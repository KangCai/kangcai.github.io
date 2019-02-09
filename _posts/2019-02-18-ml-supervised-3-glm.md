---
layout: post
title: "机器学习 · 监督学习篇 III"
subtitle: "逻辑回归与广义线性模型"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

### 一、逻辑回归

逻辑回归，即 Logistic Regression，名字的由来是因为计算过程中使用到了 Logisitic 函数，如下所示，

h_\theta(x)=\frac{1}{1+e^{\theta^T x}}
<center>
<img src="https://latex.codecogs.com/gif.latex?h_\theta(x)=\frac{1}{1&plus;e^{\theta^T&space;x}}" />
</center>

该函数也称作 sigmoid 函数，那么为什么逻辑回归会选择使用 sigmoid 函数呢？有很多文章说是因为 sigmoid有很多优秀的性质，这其实是本末倒置了，具备 sigmoid 函数这样性质的函数有很多。之所以使用 sigmoid 函数，其实是与 “逻辑回归模型对数据的先验分布假设” 直接相关，这一点在本文第二节介绍广义线性模型时再说，暂且先看逻辑回归是怎么使用 sigmoid 函数来实现分类的。对于一个二类任务，LR 公式推导如下所示，

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;(1)\&space;&&space;\left\{\begin{matrix}&space;p(y=1|x;\theta)=h_\theta(x)&space;\\&space;\&space;\&space;\&space;\&space;\&space;p(y=0|x;\theta)=1-h_\theta(x)&space;\end{matrix}\right.\\&space;(2)\&space;&&space;\Rightarrow&space;p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}&space;\\&space;(3)\&space;&&space;\Rightarrow&space;M(\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta)&space;\\&space;(4)\&space;&&space;\Rightarrow&space;M(\theta)=\prod_{i=1}^{m}(h_\theta(x^{(i)}))^y^{(i)}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}\\&space;(5)\&space;&&space;\Rightarrow&space;log(M(\theta))=\sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)}))&plus;(1-y^{(i)})log(1-h_\theta(x^{(i)}))]&space;\\&space;(6)\&space;&&space;\Rightarrow&space;L(\theta)=-log(M(\theta))=-\sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)}))&plus;(1-y^{(i)})log(1-h_\theta(x^{(i)}))]&space;\end{aligned}"/>

公式(1)中直接将自变量 x，即特征，通过 sigmoid 函数表示，这么表示的原因上文说过了会在第二节说，因变量 y 等于1的概率是 h(x)，为0的概率当然是 1-h(x)；公式(2)用一个公式涵盖了公式(1)中的两个式子；再假设每个样本互相独立，有公式(3)，即似然估计函数作为；将公式(2)代入公式(3)中，得到似然估计函数的表示(4)；为了将指数运算去掉，等式两边取 log 得到公式(5)；目的是使似然估计函数最大，而损失函数的目的是最小，所以在公式(5)上加个负号，得到公式(6)损失函数的表示。其中值得一提的是，-PlogP 是信息熵，值越大表示对当前情况越不确定，其中P是概率，而交叉熵可以用来衡量两个分布的相似度情况，假设 
 
<img src="https://latex.codecogs.com/gif.latex?p\in&space;\{y,1-y\},\&space;q\in&space;\{h_\theta(x),1-h_\theta(x)\}"/>
 
，那么我们衡量 p 和 q 的相似度就可以通过交叉熵公式来计算

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;H(p,q)&=-\sum_{i}^{&space;}p_i\&space;log\&space;q_i&space;\\&space;&=\sum_{i}^{&space;}[-y^{(i)}\&space;log\&space;h_\theta(x^{(i)})-(1-y^{(i)})\&space;log(1-h_\theta(x^{(i)})))]&space;\\&space;&=-\sum_{i}^{&space;}[y^{(i)}\&space;log\&space;h_\theta(x^{(i)})&plus;(1-y^{(i)})\&space;log(1-h_\theta(x^{(i)})))]&space;\end{aligned}" />

，跟上面逻辑回归损失函数完全一致，也就是说逻辑回归的损失函数就是交叉熵。

### 二、常见模型及其联接函数

###### 1.1 伯努利分布

###### 1.2 高斯分布

###### 1.3 二项分布

###### 1.4 泊松分布

[csdn: 机器学习分享——逻辑回归推导以及numpy的实现](https://blog.csdn.net/weixin_44015907/article/details/85306108)