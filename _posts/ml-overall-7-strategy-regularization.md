---
layout: post
title: "机器学习·总览篇 VI"
subtitle: "三要素之策略-损失函数"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)


在模型部分，机器学习的学习目标是获得假设空间（模型）的一个最优解，那么接下来如何评判解是优还是不优？

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/r&c.png"/>
</center>

如上一篇文章所说的，**策略部分就是评判“最优模型”（最优参数的模型）的准则和方法**。图1是目标函数的函数形式，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/object function.jpg"/>
</center>

如图1所示，目标函数包含了表征经验风险的代价函数和表征结构风险的正则化项，上一偏已经介绍**损失函数**，代价函数是样本集的损失函数之和；正则化是对wx+b中参数w的约束，下文将主要介绍**正则化**。


### 正则化

正则化，即Regularization

正则化是对wx+b中参数w的约束，通常按照惯例是不管b的，但如果在实际应用时在正则化函数里将b也加上的话，影响不大，效果基本没差别。

L0 L1 L2

##### L1与L2

当特征参数为1维时，L1和L2的函数曲线如图X所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/l1 and l2.png"/>
</center>

L1和L2应用于目标函数最优解求解的示意图可用图X表示，为了方便可视化，这里考虑特征参数为2维的情况，



[cnblogs: 机器学习之正则化](https://www.cnblogs.com/jianxinzhou/p/4083921.html])
[cnblogs: 机器学习中的范数规则化之（一）L0、L1与L2范数](https://www.cnblogs.com/weizc/p/5778678.html)

