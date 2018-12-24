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
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&P(|E_{in}(h)-E_{out}(h)|>\varepsilon)\\\leq\&space;&P(|E_{in}(h_1)-E_{out}(h_1)|>\varepsilon&space;\&space;\cup&space;\&space;...\&space;\cup\&space;|E_{in}(h_M)-E_{out}(h_M)|>\varepsilon)&space;\\&space;\leq&space;\&space;&&space;P(|E_{in}(h_1)-E_{out}(h_1)|>\varepsilon&space;&plus;&space;\&space;...&space;&plus;&space;\&space;P(|E_{in}(h_M)-E_{out}(h_M)|>\varepsilon)&space;\\&space;\leq&space;\&space;&&space;2Me^{-2\varepsilon^2N}&space;\end{aligned}"/>
</center>

上面这个式子的含义很重要：在假设空间 H 中，设定一个较小的 ϵ 值，任意一个假设 h ，它的样本值和期望值之间的误差概率被一个只与 ϵ、样本数 N、假设数 M 相关的值约束住。

到这里，我们可以将最开始看起来 “经验主义” 地对 learning 可行的情况定义用上面的结论改造一下，如下所示：

1. 如果备选函数集的大小 \|H\|=M ，M 有限，训练数据量 N 足够大，则对于学习算法 A 选择的任意备选函数 h，都有 <img src="https://latex.codecogs.com/gif.latex?E_{in}(h)\approx&space;E_{out}(h)" />
2. 如果 A 找到了一个备选函数，使得 <img src="https://latex.codecogs.com/gif.latex?E_{in}(h)\approx&space;0" /> ，则有很大概率使 <img src="https://latex.codecogs.com/gif.latex?E_{out}(h)\approx&space;0" />

所以将 learning 可行性的问题用上面两个结论转换一下，问题变成了：

1. 我们能否保证 E-in(h) 与 E-out(h)  足够接近？
2. 我们能否使 E-in(h)  足够小？

其中**对于第2点，能否使 E-in(h) 足够小这个问题通过合适的 “策略+算法” 可以达成**，关于这一点在前面的文章中已经解释地比较详细了（具体可参考《总览篇 VI 策略-损失函数》、《总览篇 VIII 算法-梯度下降法及其变种》、《总览篇 IX 算法-牛顿法和拟牛顿法》这3篇文章）；**对于1点，我们将在1.2节中继续分析**。
 
##### 1.2 样本损失值和总体期望损失值之间高误差的概率上界

继续上文进行分析，对于假设函数 h，我们如果能够保证 其在样本集中的损失值 与 其在总体数据集中的损失值 之间高误差的概率存在一个接近 0 的上界，那么当然就能够保证 E-in(h) 与 E-out(h) 足够接近。从上文的最后结论着手继续分析，

<center>
<img src="https://latex.codecogs.com/gif.latex?P(|E_{in}(h)-E_{out}(h)|>\varepsilon)\&space;\leq&space;\&space;2Me^{-2\varepsilon^2N}" />
</center>

对于假设空间中的备选函数，假设数 M 通常是一个无穷大的数，而ϵ、样本数 N 是一个有限的数，看起来并不存在上界。所以一个直观的思路是，能否找到一个有限的因子来替代掉上面不等式上界（右边式子）中的 M。幸运的是，存在这个因子 m-H 恒满足下面的式子：

<center>
<img src="https://latex.codecogs.com/gif.latex?P(|E_{in}(h)-E_{out}(h)|>\varepsilon)\&space;\leq&space;\&space;2\&space;m_H&space;\&space;e^{-2\varepsilon^2N}"  />
</center>

，且 m-H 是有限的，我们暂且称之为有效假设数（Effective Number of Hypotheses）。

**成长函数（Growth Function）**

继续上文分析 m-H 到底是多少。虽然假设空间中通常存在 M 个（M=无限）假设函数 h，但多个 h 之间并不是完全独立的，他们是有很大的同质性，也就是在 M 个假设中，可能有一些假设可以归为同一类。下面以二维线性假设空间为例，我们的算法要在二维空间挑选一条直线发成作为尽可能好的假设 g，用来划分一个点（N=1）的类别，虽然有无数条直线可供选择，但真正有判别意义的就两类：一类判别成正例，一类判别成反例。如下图所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/1point2lines.png"/>
</center>

即对于二维线性假设空间，当 N=1 时，<img src="https://latex.codecogs.com/gif.latex?m_H=2"/>，以此类推，可以用如下表格表示，

| N=1 | N=2 | N=3 | N=4 | 
| :----------: | :----------: | :----------: | :----------: |
| <img src="https://latex.codecogs.com/gif.latex?m_H=2"/> | <img src="https://latex.codecogs.com/gif.latex?m_H=4"/> | <img src="https://latex.codecogs.com/gif.latex?m_H=8"/>| <img src="https://latex.codecogs.com/gif.latex?m_H=14"/> | 

N 为1、2、3时，对应的有效假设数很好理解，但 N=4 时有效假设数是14而不是16，可以在纸上画一下 4 个点的两种不同的分布情况（凸轮廓和非凸轮廓）。根据以上的分析我们可以推测到一个结论，**假设空间的大小 M 虽然是很大，但在样本集 D 上，有效假设数 m-H 是有限的，m-H 随着样本数 N 变化，有效假设数 m-H 和样本 N 关系又称之为成长函数（Growth Function）**。总而言之，虽然上确界（最小上界）不能直观得出，但可以确定的是，

<center>
<img src="https://latex.codecogs.com/gif.latex?m_H(N)\leq&space;2^N" />
</center>

而且这个结论显然可以应用在任何二分类问题的假设空间，已知 **m-H 必然存在一个上界是 2^N**，则有下面的式子:

<center>
<img src="https://latex.codecogs.com/gif.latex?P(|E_{in}(h)-E_{out}(h)|>\varepsilon)\&space;\leq&space;\&space;2^{N&plus;1}e^{-2\varepsilon^2N}" />
</center>

**我们当然不会满足于成长函数是一个指数函数的结论，因为上式并不能保证概率上界（不等式的右边）是一个接近于0的值，所以我们需要进一步分析，尝试找到一个更小的上界**。上面我们具体分析了二维线性假设空间下的情况，**这里暂且不讨论二维假设空间中有效假设数的普遍结论，下文1.2节会详细讨论**，下面我们来看更多其它情况下的案例。

第一种是 “正例射线” 的假设空间，如下图所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/positive ray.png"/>
</center>

根据排列组合知识，可知 “正例射线” 的假设空间下的成长函数是

<center>
<img src="https://latex.codecogs.com/gif.latex?m_H(N)=\binom{N&plus;1}{1}=N&plus;1" />
</center>

这个假设空间的有效假设数 m-H 是 N+1，是一个关于 N 的多项式，

<center>
<img src="https://latex.codecogs.com/gif.latex?P(|E_{in}(h)-E_{out}(h)|>\varepsilon)\&space;\leq&space;\&space;2(N&plus;1)e^{-2\varepsilon^2N}" />
</center>

OK，从上式的右边可以看到 N > 0 时，上界（不等式右边）是一个随 N 递减的函数，当 N 取一个很大的值时，上界接近于0，对于这个问题，大功告成，是可学习的。

第二种是 “凸集” 的假设空间，如下图所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/convex growth function.png"/>
</center>

给定任意一种排列情况，总能找出一个凸集刚好包含了所有+1的点，并将-1的点排斥在外，所以很可惜，这种情况下的 m-H 就是 2^N。

综合以上两种情况，我们发现，**如果成长函数是指数函数，则随着 N 的增大，概率上界也急剧增加，所以我们希望成长函数是多项式**，比如 “正例射线” 的问题，就是可学习的。那么**是否有更多更普遍的成长函数是多项式的问题呢？幸运的是，答案又是肯定的**。

**打散（Shatter） 与 断点（Break Point）**

继续上文分析，为了找到更普遍的成长函数是多项式的问题，需要了解两个概念：打散（Shatter） 与 断点（Break Point）。

打散很简单，上文的 “凸集” 问题中 m-H=2^N，我们就可以称之为：N 个样本全都能被 H **打散（Shatter）**；而对于 “正例射线” 问题，当 N=1 时，m-H = 2 = 2^N，故称该问题中 1 个样本能被 H 打散。

断点和打散相关，不能被打散的样本数目就是断点，还是上面的 “正例射线” 问题，当 N>1 时，m-H 恒小于 2^N，故 N=2,3,4,... 都是断点。

总之，根据 m-H(n) = 2^n，则称 n 个样本被 H 打散； m-H(n) < 2^n，则称 n 是 H 的一个断点，通常我们只关注最小断点值，称之为 k，因为大于 k 的点都是断点。接下来就是正餐，不同的假设空间 H 的断点 k 不一样大，我们定义一个上限函数 B(N,k)，表示在断点为 k 时成长函数 m-H 的上界，这里理解断点可能有点困难，建议把它理解成一个对假设空间 H 的一个硬性要求或条件。根据上文，当然有 B(N,k) 不大于 2^N，具体有以下情况，

1. B(N,1)=1，因为 B(N,1) 表示对于 N 个样本，任意一个样本都不允许完全2分类，那么只能只有一种情况，加入任何第二种情况，总会有一个样本出现了两个分类;
2. B(N,N)=2^N-1，因为对于 N 个样本，如果没有断点，B=2^N，而在 N 处刚好有断点，那么去掉某一种情况，就可以满足了;
3. B(3,2)是多少，可以看下图，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/shattle.png"/>
</center>

故，B(3,2)=4。以此类推，可以求得所有 B(N,k），但每个都这么算好麻烦，用计算机算比较快，但难道没有一个通用的表达式吗？再次幸运的是，答案是有的。

**VC 界**

继续上文分析，根据1有 B(2,1)=1，根据2有 B(2,2)=3，根据3有B(3,2)=4。 B(3,2) 刚好等于 B(2,1)+B(2,2)，即刚好满足 B(N,k) = B(N-1,k) + B(N-1,k-1)，这不是偶然，我们可以大概推导一下： B(N,k) 比 B(N-1,k) 多了一个新样本，但断点没变，假设 B(N,k) 中有一部分（2α 个）是由 B(N-1,k) 的一部分（α 个）复制了一遍，然后分别加上新样本两类的情况；另一种是不能复制，只能加上新样本特定某一类别的那部分（β 个）。由于 B(N,k) 中存在 2α 个复制出的 case 保证了断点是 k，这意味着把新样本去掉后，原始 α 个 case 需要保证 N-1 个样本时断点是 k-1，否则 2α 个 N 样本的 case 的断点必然不是 k。根据以上推导可以得到

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&B(N,k)=2\alpha&plus;\beta\\&space;&B(N,k-1)=\alpha&plus;\beta\\&space;&B(N-1,k-1)=\alpha\\&space;\Rightarrow&space;B(N,k)&space;&&space;=&space;B(N-1,k)&space;&plus;&space;B(N-1,k-1)&space;\end{aligned}"/>
</center>

以上推导过程可以不用太在意，关键是这个递推公式的结论十分重要，根据以上递推公式，可以通过数学归纳法求 B(N,k) 的表达式，熟悉的高考题味道 :) ，推导过程省略，结论就是，

<center>
<img src="https://latex.codecogs.com/gif.latex?B(N,k)=\sum_{i=0}^{k-1}\binom{N}{i}"/>
</center>

当 N >> k时，有如下近似：

<center>
<img src="https://latex.codecogs.com/gif.latex?B(N,k)=\sum_{i=0}^{k-1}\binom{N}{i}\approx&space;\binom{N}{k-1}\approx&space;\lambda&space;N^{k-1}"/>
</center>

所以
我们先看一下这个式子的意义，如果假设空间存在有限的断点（Break Point），那么 m_H(N) 会被最高幂次为 k–1 的多项式上界给约束住；。随着N的逐渐增大，指数式的下降会比多项式的增长更快，所以此时VC Bound是有限的。更深的意义在于，N足够大时，对H中的任意一个假设h，Ein(h)都将接近于Eout(h)，这表示学习可行的第一个条件是有可能成立的。

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

https://blog.csdn.net/xiong452980729/article/details/52122081

[wiki: Hoeffding不等式](https://zh.wikipedia.org/wiki/Hoeffding%E4%B8%8D%E7%AD%89%E5%BC%8F)
[cnblogs: 解读机器学习基础概念：VC维的来龙去脉](https://www.cnblogs.com/gkwang/p/5046188.html)
[csdn: 详解机器学习中的VC维](https://blog.csdn.net/baimafujinji/article/details/44856089)
[Machine Learning - VC Dimension and Model
Complexity](https://www.cs.cmu.edu/~epxing/Class/10701/slides/lecture16-VC.pdf)
[PAC Learning and The VC Dimension](https://www.cs.cmu.edu/~ggordon/10601/recitations/rec09/pac_learning.pdf)