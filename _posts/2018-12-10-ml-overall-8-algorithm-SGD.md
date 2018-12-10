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

> 机器学习最后一个要素-算法，准确来说是优化算法，特别是数值优化（Numerical Optimization）或凸优化（Convex Optimization），属于运筹学（最优化理论）的一部分。机器学习中广泛使用的凸优化方法主要分为梯度下降法和拟牛顿法，这两类方法都派生出大量的变种，本文将介绍梯度下降法及其变种。	

> 文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)

### 一、运筹学和凸优化

##### 1.1 运筹学

运筹学（最优化理论）目前在机器学习领域应用非常广泛，因为机器的学习过程简单来说，主要做的就是优化问题，先初始化一下权重参数，然后利用优化方法来优化这个权重，直到目标函数不再下降、准确率不再上升，迭代停止。

那到底什么是最优化问题？它的一般形式是：

（公式）min f(x) s.t x 属于 X

可以看到，之前介绍的机器学习第二要素——策略的目标函数，

（公式）

就属于最优化问题。在运筹学中，根据 f(x) 和 X 形式的不同，分为以下分支：

* 线性规划。当目标函数 f 是线性函数而且变量约束 X 是由线性等式函数或不等式函数来确定的，这类问题为线性规划。
* 整数规划。当 X 包含对变量的整数变量约束，这类问题为整数规划。
* 非线性规划。目标函数 f 或变量约束 X 中包含非线性函数，这类问题为非线性规划。
* 几何规划。非线性规划的一个分支，	
* 网络流优化。当 X 包含连同赋权有向图约束，这类问题为网络流优化，物流、电网、通讯网络甚至是滴滴拼车等调度规划应用都属于该范畴。
* 随机规划。x 中部分或全部变量是随机变量，需要分析随机变量的概率分布，以及各随机变量的自相关和互相关，求解方法比确定性规划复杂得多，这类问题是随机规划。
* 鲁棒优化。

##### 1.2 凸优化

运筹学中一类重要的优化问题是凸优化，**凸优化 = 凸集 + 凸函数 + 存在一阶导为0**，其中，


（图）凸优化定义
<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/convex set1.png"/>
</center>

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/convex function.png"/>
</center>

凸优化具有以下特性：

1. **局部最优解 = 全局最优解，是目前技术比较确定可以求解的非线性规划方式，且效率高**
2. **许多非凸问题通过一定的手段，转换成凸问题求解**。要么等价地化归为凸问题，要么用凸问题去近似、逼近。典型的如几何规划、整数规划，它们本身是非凸的，但是可以借助凸优化手段去解，这就极大地扩张了凸优化的应用范围
3. **在非凸优化中，凸优化同样起到很重要的作用**，
    + 当你要解决一个非凸优化问题时，可以先试图建立一个简化多凸优化模型，解出来以后作为非凸问题的一个起始点。
	+ 很多非凸优化问题的启发式算法的基础都是基于凸优化 
	+ 你可以先建立非凸优化的松弛问题，使用凸优化算法求解，作为非凸优化问题的上限或下限

##### 1.3 机器学习中的凸优化

机器学习中涉及到的运筹学主要包含线性规划、部分非线性规划问题，优化手段主要就是凸优化：

1. **“线性规划”与“目标函数与约束函数都是凸函数的非线性规划“，属于凸优化**。这类问题在机器学习中十分常见：比如**线性回归（Linear Regression）** 是线性规划，属于凸优化；比如常见的**支持向量机（SVM）** 和**逻辑回归（Logistic Regression）** 都是目标函数与条件函数都是凸函数的非线性规划，属于凸优化。
2. **对于非凸优化，通过一定手段，可以等价化归为凸问题**。比如对于变量要求是整数、目标函数与约束函数都是凸函数的整型规划，可以通过凸松弛手段转换成凸优化问题。另外，对于非凸的优化问题，我们可以将其转化为对偶问题，对偶函数一定是凹函数，但是这样求出来的解并不等价于原函数的解，只是原函数的一个确下界，关于这一点可以看前一篇介绍正则化时说到的拉格朗日乘子法，不论原问题是不是一个凸优化问题，只要满足KKT条件，转换成的对偶问题就是一个凸优化问题，可以用凸优化方法求解。
3. **对于非凸优化，在机器学习中直接用凸优化的手段去求最优解效果也不错**，这跟做Optimization时的结论完全相反，这个现象与机器学习中各种正则化和随机化方法有关，从这里可见机器学习的工程性。比如**深度学习**的目标函数就是高度非凸目标函数，然而我们通常也是直接上凸优化方法。

机器学习中常用的凸优化算法分为两大类：梯度下降法（Gradient Descent）和拟牛顿法（）。

##### 1.4 梯度下降法和拟牛顿法的比较

！！！https://www.zhihu.com/question/46441403
https://www.cnblogs.com/8335IT/p/5809495.html 牛顿法和梯度下降法对比
http://blog.sina.com.cn/s/blog_1442877660102wru5.html

因为两类方法的变种很多，所以拆分成两篇文章分别进行介绍，本文第二节将介绍梯度下降法及其变种，下一篇文章介绍拟牛顿法。

### 二、梯度下降法及其变种

##### 2.1 梯度下降法（Gradient Descent, GD）

想象你在一个山峰上，在不考虑其他因素的情况下，你要如何行走才能最快的下到山脚？当然是选择最陡峭的地方，这也是梯度下降法的核心思想：它通过每次在当前梯度方向（最陡峭的方向）向前迈一步，来逐渐逼近函数的最小值。假设在第n次迭代中，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta_n&space;=&space;\theta_{n-1}&plus;\Delta&space;\theta"/>
</center>

我们将损失函数在 θn-1 处进行一阶泰勒展开：

<center>
<img src="https://latex.codecogs.com/gif.latex?L(\theta_n)=L(\theta_{n-1}&plus;\Delta\theta)\approx&space;L(\theta_{n-1})&plus;L^{'}(\theta_{n-1})\Delta\theta" />
</center>

为了使L(θ)迭代后趋于更小的值，可以假定以下所示第一行条件，

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\Delta\theta&=-\alpha&space;L'(\theta_{n-1}),&space;\alpha>0&space;\\&space;\text{&space;s.t.&space;}&space;L(\theta_n)&=L(\theta_{n-1})-\alpha&space;L^{'}(\theta_{n-1})^2&space;\\&space;&\leq&space;L(\theta_{n-1})&space;\end{aligned}"/>
</center>

故迭代方式可以用如下公式表示，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta_n=\theta_{n-1}-\alpha&space;L'(\theta_{n-1})"  />
</center>

目标函数关于模型参数的一阶导L′(θn)在不同的方法中都会使用到，为了方便起见，后文用gn来代替L′(θn)，如下所示，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta_n=\theta_{n-1}-\alpha&space;g_{n-1}"/>
</center>

梯度下降法根据每次求解损失函数L带入的样本数，可以分为：全量梯度下降（计算所有样本的损失），批量梯度下降（每次计算一个batch样本的损失）和随机梯度下降（每次随机选取一个样本计算损失）。现在所说的SGD（Stochastic Gradient Descent，随机梯度下降）多指Mini-batch Gradient Descent（批量梯度下降）。

**优点：求一阶导，操作简单，计算量小**，在损失函数是凸函数的情况下能够保证收敛到一个较好的全局最优解。

**缺点：α是个定值（在最原始的版本），收敛的好坏受它的直接影响**，过小会导致收敛太慢，过大会导致震荡而无法收敛到最优解。对于非凸问题，只能收敛到局部最优，并且没有任何摆脱局部最优的能力（一旦梯度为0就不会再有任何变化）。

##### 2.2 GD迭代方向的改进

GD的迭代公式如下所示，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta_n=\theta_{n-1}-\alpha&space;g_{n-1}"/>
</center>

一类常见的变种是对迭代方向 g 进行改进，比如下面三种，

**Momentum**

SGD具有收敛不稳定：在SGD中，参数 θ 的迭代变化量都只是与当前梯度正相关，这会导致在迭代过程中，θ在最优解附近以较大幅度震荡无法收敛。Momentum能在一定程度上缓解SGD收敛不稳定的问题，它的思想就是模拟物体运动的惯性：当我们跑步时转弯，我们最终的前进方向是由我们之前的方向和转弯的方向共同决定的。**Momentum在每次更新时，保留一部分上次的更新方向**： 

（公式）

**优点：**1. 由于在迭代时考虑了上一次 θ 的迭代变化量，所以在最优解附近通常具有梯度因方向相反而抵消的效果，从而在一定程度上缓解SGD的收敛不稳定问题；2. 除此之外，Momentum还具有一定的摆脱局部最优的能力，如果上一次梯度较大，而当前梯度为0时，仍可能按照上次迭代的方向冲出局部最优点。对于第一点直观上的理解是，它可以让每次迭代的“掉头方向不是那么大“，左图为SGD，右图为Momentum。

（图 from An overview of gradient descent optimization algorithms）

**缺点：**这里又多了另外一个超参数 ρ 需要我们设置，它的选取同样会影响到结果。

**Nesterov Momentum**

Nesterov Momentum又叫做Nesterov Accelerated Gradient（NAG），是基于Momentum的加速算法。 Momentum的 θ 迭代向量的计算其实是把上一次迭代向量作为本次预期迭代向量，在此基础上叠加上本次梯度。NAG认为这样的迭代不合理，迭代速度也不够快，NAG认为：既然都把上一次迭代向量作为本次预期向量了，就应该预期到底，应该将当前梯度替换成预期梯度，即当前θ加上上一次迭代向量之后的新θ所在点的函数梯度，

（公式）
<img src="https://latex.codecogs.com/gif.latex?\Delta\theta_n&space;=\rho\Delta\theta_{n-1}&plus;&space;g(\theta_{n-1}-\alpha&space;\Delta\theta_{n-1})"/>

上述公式通过微分的思想可以做如下转换，

（公式）

分析上式，其实可以看到它实际使用到了二阶导数的信息，二阶导数>0即一阶导单调递增，也即g′n−1>g′n−2，因此可以加快收敛，从这里可以看到这种做法的内在理论：如果这次的梯度比上次大，那么我们有理由认为梯度还会继续变大，所以当前迈的步子可以更大一些。

**共轭梯度法（Conjugate Gradient）**

http://www.docin.com/p-611887237.html

就是把目标函数分成许多方向，然后不同方向分别求出极值在综合起来。

不同于上述算法对迭代方向进行改进，后面这些算法主要研究沿着梯度方向走多远的问题，也即如何选择合适的学习率α。

##### 2.3 GD学习率的改进

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\Delta\theta&=-\alpha&space;L'(\theta_{n-1}),&space;\alpha>0&space;\\&space;\text{&space;s.t.&space;}&space;L(\theta_n)&=L(\theta_{n-1})-\alpha&space;L^{'}(\theta_{n-1})^2&space;\\&space;&\leq&space;L(\theta_{n-1})&space;\end{aligned}"/>

**Adagrad**

即adaptive gradient，自适应梯度法。它通过记录每次迭代过程中的前进方向和距离，从而使得针对不同问题，有一套自适应调整学习率的方法

<img src="https://latex.codecogs.com/gif.latex?\alpha=\frac{1}{\sqrt{\sum_{i=1}^{n-1}g_{i}&plus;\epsilon}}\alpha_0" />

优点：解决了SGD中学习率不能自适应调整的问题 
缺点：学习率单调递减，在迭代后期可能导致学习率变得特别小而导致收敛及其缓慢。同样的，我们还需要手动设置初始α


**Adadelta**


<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha=\frac{1}{\vert&space;diag(H_n)\vert}*\frac{(\sum_{i=n-t}^{n-1}g_{i})^2}{\sum_{i=n-t}^{n-1}g^{2}_{i}}\alpha_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha=\frac{1}{\vert&space;diag(H_n)\vert}*\frac{(\sum_{i=n-t}^{n-1}g_{i})^2}{\sum_{i=n-t}^{n-1}g^{2}_{i}}\alpha_0" title="\alpha=\frac{1}{\vert diag(H_n)\vert}*\frac{(\sum_{i=n-t}^{n-1}g_{i})^2}{\sum_{i=n-t}^{n-1}g^{2}_{i}}\alpha_0" /></a>


**Adadelta**


Adadelta在《ADADELTA: An Adaptive Learning Rate Method 》一文中提出，它解决了Adagrad所面临的问题。

<img src="https://latex.codecogs.com/gif.latex?RMS[g]_{n}=\sqrt{E[g^2]_n&plus;\epsilon}"  />

<img src="https://latex.codecogs.com/gif.latex?E[g^2]_n=\rho&space;E[g^2]_{n-1}&plus;(1-\rho)g_n^2" />

<img src="https://latex.codecogs.com/gif.latex?\alpha=\frac{RMS[\Delta\theta]_{n-1}}{RMS[g]_n}\alpha_0"  />


这里ρρ为小于1的正数，随着迭代次数的增加，同一个会因为累乘一个小于1的数而逐渐减小，即使用了一种自适应的方式，让距离当前越远的梯度的缩减学习率的比重越小。分子是为了单位的统一性，其实上述的算法中，左右的单位是不一致的，为了构造一致的单位，我们可以模拟牛顿法（一阶导\二阶导），它的单位是一致的，而分子就是最终推导出的结果，具体参考上面那篇文章。这样，也解决了Adagrad初始学习率需要人为设定的问题。


**RMSprop**


其实它就是Adadelta，这里的RMS就是Adadelta中定义的RMS，也有人说它是一个特例，ρ=0.5的Adadelta。


**Adam**


Adam是Momentum和Adaprop的结合体，我们先看它的更新公式

<img src="https://latex.codecogs.com/gif.latex?E[g^2]_n=\rho&space;E[g^2]_{n-1}&plus;(1-\rho)g_n^2" />

<img src="https://latex.codecogs.com/gif.latex?E[g]_n=\phi&space;E[g]_{n-1}&plus;(1-\phi)g_n^2" />

<img src="https://latex.codecogs.com/gif.latex?\bar{E[g^2]_n}=\frac{E[g^2]_n}{1-\rho^n}"  />

<img src="https://latex.codecogs.com/gif.latex?\bar{E[g]_n}=\frac{E[g]_n}{1-\phi^n}" />

<img src="https://latex.codecogs.com/gif.latex?\alpha=\frac{\bar{E[g]_n}}{\sqrt{\bar{E[g^2]_n}}&plus;\epsilon}\alpha_0"/>




它利用误差函数的一阶矩估计和二阶矩估计来约束全局学习率。 
优点：结合Momentum和Adaprop，稳定性好，同时相比于Adagrad，不用存储全局所有的梯度，适合处理大规模数据 
一说，adam是世界上最好的优化算法，不知道用啥时，用它就对了。
《Adam: A Method for Stochastic Optimization》

**Adamax**

<center>
<img src="https://latex.codecogs.com/gif.latex?E[g^2]_n=\max(\vert&space;g_n\vert,\rho&space;E[g^2]_{n-1})"  />
</center>

<center>
<img src="https://latex.codecogs.com/gif.latex?\alpha=\frac{\bar{E[g]_n}}{E[g^2]_n+\epsilon} \alpha_0"/>
</center>

[Convex Optimization, by S. Boyd and L. Vandenberghe]()
[jianshu: 运筹学(最优化理论)主要概念区分](https://www.jianshu.com/p/c8f5e1631b29)
[zhihu: 运筹学（最优化理论）如何入门？](https://www.zhihu.com/question/22686770)
[csdn: 关于线性规划、非线性规划与凸优化](https://blog.csdn.net/weixin_37589896/article/details/78712955)
[csdn: 最全的机器学习中的优化算法介绍](https://blog.csdn.net/qsczse943062710/article/details/76763739)
