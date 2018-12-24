---
layout: post
title: "机器学习 · 总览篇 X"
subtitle: "可学习性 & VC维"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)



机器学习主要研究的是怎么去学习解决一个问题，这里面包含了一个隐含的前提条件：学习解决的问题必须是可学习的问题。那么怎么去判定一个问题的可学习性呢？PCA Learning 就是关于机器学习可学习性的一个完善的解释理论。PAC learning，全称是 Probably approximately correct learning，中文直译名字比较拗口，叫 概率近似正确学习，解释这个名字：

1. 首先，Approximately Correct（近似正确）就是指学出的模型的误差比较小（误差被限制住），因为实现零误差（Absolutely Correct）是非常困难并且通常没有必要的，所以这里考虑的是 Approximately Correct；
2. 其次，由于随机性的存在，我们只能从概率上保证 Approximately Correct 的可能性是很大的（存在一个概率下界）。

以上这就是 PAC Learning 的名称由来。Leslie Valiant 于1984年提出 PAC Learning，也主要因为该理论获得2010年图灵奖，可见该理论对机器学习的重要性。 PAC Learning 可以看做是机器学习的数学分析框架，它将计算复杂度理论引入机器学习，描述了机器学习的有限假设空间的可学习性，无限空间的VC维相关的可学习性等问题。

下面将从 可学习性 和 VC维 两个部分对 PAC Learning 理论进行介绍。

https://blog.csdn.net/csshuke/article/details/52221873

### 一、可学习性

在什么情况下 learning 是可行的？以机器学习实际应用的角度来看，需要具备以下三个条件，

1. 模型不能过于复杂，数据量需要足够大，即模型的复杂程度不能远高于数据量的支撑
2. 合适的最优化方法，即让 目标函数值接近0 的求参算法

这三个条件看起来属于 “经验主义”，那有没有更加准确的数学程式化定义？

##### 1.1 Hoeffding不等式 与 机器学习

为了解答上面的问题，需要从 Hoeffding不等式 说起，Hoeffding不等式 是关于一组随机变量均值的概率不等式。 如果 X1,X2,⋯,Xn 为一组独立同分布的参数为 p 的伯努利分布随机变量，n为随机变量的个数。定义这组随机变量的均值为：

<center>
<img src="https://latex.codecogs.com/gif.latex?\bar{X}=\frac{X_1&plus;X_2&plus;...&plus;X&plus;n}{n}" />
</center>

那么对于任意 δ>0, Hoeffding不等式 可以表示为

<center>
<img src="https://latex.codecogs.com/gif.latex?P(|\bar{X}-E(\bar{X})|\geq&space;\delta)\&space;\leq&space;\&space;2e^{-2\delta^2n^2}" />
</center>

Hoeffding不等式 可以直接应用到一个 抽球颜色 的统计推断问题上：我们从罐子里抽球，希望估计罐子里红球和绿球的比例，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/bin_sample1.png"/>
</center>

如果对 总览篇III 一文中涉及的统计推断方法还记得的话，知道这个问题根据 频率学派 和 贝叶斯学派 的差别有两个不同的答案，频率学派给出的答案就是 总体的期望 μ 就等于样本期望 ν，这里对两个学派就不再次进行解释了，只讨论频率学派给出的答案的准确性。直觉上，如果我们有更多的样本，即抽出更多的球，总体的期望 μ 确实越接近样本期望 ν；事实上，这里可以用 Hoeffding不等式 量化地表示接近情况，如果抽球样本数维 N，则如下：

<center>
<img src="https://latex.codecogs.com/gif.latex?P(|v-\mu|>\varepsilon&space;)\&space;\leq&space;\&space;2e^{-2\varepsilon^2N}"/>
</center>

再进一步到机器学习的问题上，机器学习的过程可以程式化表示为：通过算法 A，在机器学习方法的假设空间 H 中，根据样本集 D，选择最好的假设作为 g，选择标准是使 g 近似与理想的方案 f，其中，H 可以是一个函数（此时是非概率模型），也可以是一个分布（此时是概率模型），g 和 f 属于 H。类似于上面 “抽球” 的例子，可以通过样本集的经验损失（expirical loss ） <img src="https://latex.codecogs.com/gif.latex?E_{in}(h)" title="E_{in}(h)" /> ，即 in-sample error，来推测总体的期望损失（expected loss） <img src="https://latex.codecogs.com/gif.latex?E_{out}(h)"/>。对于假设空间 H 中一个任意的备选函数 h，基于 Hoeffding不等式，我们得到下面的式子：

<center>
<img src="https://latex.codecogs.com/gif.latex?P(|E_{in}(h)-E_{out}(h)|>\varepsilon)\leq\&space;2e^{-2\varepsilon^2N}" />
</center>

那么对于整个假设空间 H，假设存在 M 个 h，则可以推导出下面的式子：

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&P(|E_{in}(h_1)-E_{out}(h_1)|>\varepsilon&space;\&space;\cup&space;\&space;...\&space;\cup\&space;|E_{in}(h_m)-E_{out}(h_m)|>\varepsilon)&space;\\&space;\leq&space;\&space;&&space;P(|E_{in}(h_1)-E_{out}(h_1)|>\varepsilon&space;&plus;&space;\&space;...&space;&plus;&space;\&space;P(|E_{in}(h_m)-E_{out}(h_m)|>\varepsilon)&space;\\&space;\leq&space;\&space;&&space;2Me^{-2\varepsilon^2N}&space;\end{aligned}"/>
</center>

上面这个式子的含义很重要：在假设空间 H 中，设定一个较小的 ϵ 值，任意一个假设 h ，它的样本值和期望值的误差被一个只与 ϵ、样本数 N、假设数 M 相关的值约束住。

到这里，我们可以将最开始看起来 “经验主义” 地对 learning 可行的情况定义用上面的结论改造一下，如下所示：

1. 如果备选函数集的大小 |H|=M ，M 有限，训练数据量 N 足够大，则对于学习算法 A 选择的任意备选函数 h，都有 <img src="https://latex.codecogs.com/gif.latex?E_{in}(h)\approx&space;E_{out}(h)" />
2. 如果 A 找到了一个备选函数，使得 <img src="https://latex.codecogs.com/gif.latex?E_{in}(h)\approx&space;0" /> ，则有很大概率使 <img src="https://latex.codecogs.com/gif.latex?E_{out}(h)\approx&space;0" />

所以将 learning 可行性的问题用上面两个结论转换一下，问题变成了：

1. 我们能否保证 <img src="https://latex.codecogs.com/gif.latex?E_{out}(h)" /> 与 <img src="https://latex.codecogs.com/gif.latex?E_{in}(h)" />  足够接近？
2. 我们能否使 <img src="https://latex.codecogs.com/gif.latex?E_{in}(h)" />  足够小？

其中**对于第2点，能否使<img src="https://latex.codecogs.com/gif.latex?E_{in}(h)" />  足够小这个问题通过合适的 “策略+算法” 可以达成**，关于这一点在前面的文章中已经解释地比较详细了（具体可参考《总览篇 VI 策略-损失函数》、《总览篇 VIII 算法-梯度下降法及其变种》、《总览篇 IX 算法-牛顿法和拟牛顿法》这3篇文章）；**对于1点，我们将在1.2节中继续分析**。
 
##### 1.2 假设函数 h 在样本集和总体集的误差上界



**成长函数（Growth Function）**

Effective Number of Hypotheses（假设函数的有效值）
Effective Number of Lines（有效线的数量）几何理解

https://blog.csdn.net/huang1024rui/article/details/47165887

**打散（Shatter） 与 断点（Break Point）**

https://blog.csdn.net/xiong452980729/article/details/52122081

**VC 界**

https://blog.csdn.net/xiong452980729/article/details/52122081

关于这个公式的数学推导，我们可以暂且不去深究。我们先看一下这个式子的意义，如果假设空间存在有限的break point，那么m_H(2N)会被最高幂次为k–1的多项式上界给约束住。随着N的逐渐增大，指数式的下降会比多项式的增长更快，所以此时VC Bound是有限的。更深的意义在于，N足够大时，对H中的任意一个假设h，Ein(h)都将接近于Eout(h)，这表示学习可行的第一个条件是有可能成立的。

### 二、VC 维

定义 XXXX

将N个点进行分类，如分成两类，那么可以有2^N种分法，即可以理解成有2^N个学习问题。若存在一个假设H，能准确无误地将2^N种问题进行分类。那么这些点的数量N，就是H的VC维。

用 shatter 来解释就是，一个假设空间 H 的 VC 维，是这个 H 最多能够 shatter 掉的点的数量

##### 一些问题


**二维线性分类器**

二维线性分类器的 VC 维为什么是3，而不能分散4个样本的集合。

N维实数空间中线性分类器和线性实函数的VC维是 n+1


对于一个给定的分类器或者Hypothsis，还如何确定VC维呢？一个不好的消息是，对于非线性分类器，VC维非常难于计算，在学术研究领域，这仍然是一个有待回答的开放性问题。一个好消息是，对于线性分类器，VC维是可以计算的，所以下面我们主要讨论线性分类器的VC维。

##### 相关结论

据此，大致可以总结出如下结论：通常我们可以通过改进Training error的方法来试图改善Test error，但是Improving on training error not always improves test error。所以现在的问题是有没有一个能够将training error和test error联系起来的公式？答案是肯定的。The VC dimension can predict a probabilistic upper bound on the test error of a classification model。Vapnik proved that the probability of the test error distancing from an upper bound (on data that is drawn i.i.d. from the same distribution as the training set) is given by:

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/vc error relation.png"/>
</center>

where D is the VC dimension of the classification model, and N is the size of the training set (restriction: this formula is valid when D远远小于N。 When D is larger, the test-error may be much higher than the training-error. This is due to 过拟合)。

研究人员通过分析得出结论：经验风险最小化学习过程一致的必要条件是函数集的VC维有限，且这时的收敛速度是最快的

可学习性是从问题的角度出发，去探讨问题可以被学习解决方案所学习的可能性，注意，这里“问题”是讨论的基本点。

[wiki: Hoeffding不等式](https://zh.wikipedia.org/wiki/Hoeffding%E4%B8%8D%E7%AD%89%E5%BC%8F)
[cnblogs: 解读机器学习基础概念：VC维的来龙去脉](https://www.cnblogs.com/gkwang/p/5046188.html)
[csdn: 详解机器学习中的VC维](https://blog.csdn.net/baimafujinji/article/details/44856089)
[Machine Learning - VC Dimension and Model
Complexity](https://www.cs.cmu.edu/~epxing/Class/10701/slides/lecture16-VC.pdf)
[PAC Learning and The VC Dimension](https://www.cs.cmu.edu/~ggordon/10601/recitations/rec09/pac_learning.pdf)