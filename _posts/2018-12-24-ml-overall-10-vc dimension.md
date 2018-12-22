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

Leslie Valiant提出概率近似正确学习（Probably approximately correct learning，PAC learning），是机器学习的数学分析的框架，它将计算复杂度理论引入机器学习，描述了机器学习的有限假设空间的可学习性，无限空间的VC维相关的可学习性等问题。

读懂 https://www.cnblogs.com/gkwang/p/5046188.html，PCA和VC维的概念

### 一、VC 维的含义

定义 XXXX

将N个点进行分类，如分成两类，那么可以有2^N种分法，即可以理解成有2^N个学习问题。若存在一个假设H，能准确无误地将2^N种问题进行分类。那么这些点的数量N，就是H的VC维。


##### 一些问题


**二维线性分类器**

二维线性分类器的 VC 维为什么是3，而不能分散4个样本的集合。

N维实数空间中线性分类器和线性实函数的VC维是 n+1


对于一个给定的分类器或者Hypothsis，还如何确定VC维呢？一个不好的消息是，对于非线性分类器，VC维非常难于计算，在学术研究领域，这仍然是一个有待回答的开放性问题。一个好消息是，对于线性分类器，VC维是可以计算的，所以下面我们主要讨论线性分类器的VC维。

### 二、VC 维

据此，大致可以总结出如下结论：通常我们可以通过改进Training error的方法来试图改善Test error，但是Improving on training error not always improves test error。所以现在的问题是有没有一个能够将training error和test error联系起来的公式？答案是肯定的。The VC dimension can predict a probabilistic upper bound on the test error of a classification model。Vapnik proved that the probability of the test error distancing from an upper bound (on data that is drawn i.i.d. from the same distribution as the training set) is given by:

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/vc error relation.png"/>
</center>

where D is the VC dimension of the classification model, and N is the size of the training set (restriction: this formula is valid when D远远小于N。 When D is larger, the test-error may be much higher than the training-error. This is due to 过拟合)。

研究人员通过分析得出结论：经验风险最小化学习过程一致的必要条件是函数集的VC维有限，且这时的收敛速度是最快的

[cnblogs: 解读机器学习基础概念：VC维的来龙去脉](https://www.cnblogs.com/gkwang/p/5046188.html)
[csdn: 详解机器学习中的VC维](https://blog.csdn.net/baimafujinji/article/details/44856089)
[Machine Learning - VC Dimension and Model
Complexity](https://www.cs.cmu.edu/~epxing/Class/10701/slides/lecture16-VC.pdf)
[PAC Learning and The VC Dimension](https://www.cs.cmu.edu/~ggordon/10601/recitations/rec09/pac_learning.pdf)