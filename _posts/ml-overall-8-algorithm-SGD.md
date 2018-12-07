---
layout: post
title: "机器学习 · 总览篇 VIII"
subtitle: "三要素之算法 - 梯度下降法及其变种"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)

在机器学习中，有很多的问题并没有解析形式的解，或者有解析形式的解但是计算量很大（譬如，超定问题的最小二乘解），对于此类问题，通常我们会选择采用一种迭代的优化方式进行求解。 


这些常用的优化算法包括：梯度下降法（Gradient Descent），共轭梯度法（Conjugate Gradient），Momentum算法及其变体，AdaGrad，Adadelta，RMSprop，Adam及其变体，Nadam。

牛顿法与拟牛顿法，DFP法，BFGS法，L-BFGS法
高斯-牛顿法（GD）、莱文贝格-马夸特方法（LM）、通常是针对非线性最小二乘问题，我们将在《机器学习·有监督学习篇》中的某一篇对解决最小二乘问题的各种算法进行详细的介绍。

！！！https://www.zhihu.com/question/46441403
https://www.cnblogs.com/8335IT/p/5809495.html 牛顿法和梯度下降法对比
http://blog.sina.com.cn/s/blog_1442877660102wru5.html

### 梯度下降法（Gradient Descent）

SGD的优缺点
优点：操作简单，计算量小，在损失函数是凸函数的情况下能够保证收敛到一个较好的全局最优解。 
缺点：
α是个定值（在最原始的版本），它的选取直接决定了解的好坏，过小会导致收敛太慢，过大会导致震荡而无法收敛到最优解。

对于非凸问题，只能收敛到局部最优，并且没有任何摆脱局部最优的能力（一旦梯度为0就不会再有任何变化）。 
PS：对于非凸的优化问题，我们可以将其转化为对偶问题，对偶函数一定是凹函数，但是这样求出来的解并不等价于原函数的解，只是原函数的一个确下界

### Momentum

### Nesterov Momentum
Nesterov Momentum又叫做Nesterov Accelerated Gradient（NAG），是基于Momentum的加速算法。 

### 共轭梯度法（Conjugate Gradient）

http://www.docin.com/p-611887237.html

不同于上述算法对前进方向进行选择和调整，后面这些算法主要研究沿着梯度方向走多远的问题，也即如何选择合适的学习率α。

### Adagrad

即adaptive gradient，自适应梯度法。它通过记录每次迭代过程中的前进方向和距离，从而使得针对不同问题，有一套自适应调整学习率的方法

优点：解决了SGD中学习率不能自适应调整的问题 
缺点：学习率单调递减，在迭代后期可能导致学习率变得特别小而导致收敛及其缓慢。同样的，我们还需要手动设置初始α

### Adadelta

Adadelta在《ADADELTA: An Adaptive Learning Rate Method 》一文中提出，它解决了Adagrad所面临的问题。

这里ρρ为小于1的正数，随着迭代次数的增加，同一个E[g2]iE[g2]i会因为累乘一个小于1的数而逐渐减小，即使用了一种自适应的方式，让距离当前越远的梯度的缩减学习率的比重越小。分子是为了单位的统一性，其实上述的算法中，左右的单位是不一致的，为了构造一致的单位，我们可以模拟牛顿法（一阶导\二阶导），它的单位是一致的，而分子就是最终推导出的结果，具体参考上面那篇文章。这样，也解决了Adagrad初始学习率需要人为设定的问题。

### RMSprop

其实它就是Adadelta，这里的RMS就是Adadelta中定义的RMS，也有人说它是一个特例，ρ=0.5的Adadelta，且分子α，即仍然依赖于全局学习率。

### Adam

Adam是Momentum和Adaprop的结合体，我们先看它的更新公式

它利用误差函数的一阶矩估计和二阶矩估计来约束全局学习率。 
优点：结合Momentum和Adaprop，稳定性好，同时相比于Adagrad，不用存储全局所有的梯度，适合处理大规模数据 
一说，adam是世界上最好的优化算法，不知道用啥时，用它就对了。
《Adam: A Method for Stochastic Optimization》

### Nadam和NadaMax
Nadam是带有NAG的adam：
每次迭代的ϕ都是不同的，如果参考Adamax的方式对二阶矩估计做出修改，我们可以得到NadaMax， 
详见：《Incorporating Nesterov Momentum intoAdam》


[csdn: 最全的机器学习中的优化算法介绍](https://blog.csdn.net/qsczse943062710/article/details/76763739)
