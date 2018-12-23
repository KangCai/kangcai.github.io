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


https://www.cnblogs.com/gkwang/p/5046188.html
通俗概念

##### 1.1 Hoeffding不等式 与 机器学习

https://www.cnblogs.com/gkwang/p/5046188.html

如果备选函数集的大小|H|=M，M有限，训练数据量N足够大，则对于学习算法A选择的任意备选函数h，都有  E-out(h)≈E-in(h)

如果A找到了一个备选函数，使得E-in(h)≈0，则有很大概率E-out(h)≈0

 我们能否保证E-out(h)与E-in(h)足够接近？
 我们能否使E-in(h)足够小？
 
##### 1.2 假设函数的上界

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

[cnblogs: 解读机器学习基础概念：VC维的来龙去脉](https://www.cnblogs.com/gkwang/p/5046188.html)
[csdn: 详解机器学习中的VC维](https://blog.csdn.net/baimafujinji/article/details/44856089)
[Machine Learning - VC Dimension and Model
Complexity](https://www.cs.cmu.edu/~epxing/Class/10701/slides/lecture16-VC.pdf)
[PAC Learning and The VC Dimension](https://www.cs.cmu.edu/~ggordon/10601/recitations/rec09/pac_learning.pdf)