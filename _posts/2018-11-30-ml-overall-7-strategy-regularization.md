---
layout: post
title: "机器学习·总览篇 VII"
subtitle: "三要素之策略-正则化"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

> 上一篇文章和本文主要介绍三要素的第二要素-策略：上一篇文章中已经介绍了策略中平衡经验风险和结构风险的正则化系数，衡量经验风险的代价函数和损失函数；本文继续介绍最后一个重要的内容，衡量结构风险的正则化项。

> 文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)


在上一篇文章中提到过，**策略部分就是评判“最优模型”（最优参数的模型）的准则和方法**，图1是目标函数的函数形式，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/object function 1.png"/>
</center>

如图1所示，目标函数包含了表征经验风险的代价函数和表征结构风险的正则化项，**上篇文章已经介绍了正则化系数和损失函数**，代价函数是样本集的损失函数之和；正则化是对wx+b中参数w的约束，**下文将主要介绍正则化**。


### 一、正则化

介绍正则化，首先回答下面三个问题：

**1) 机器学习中正则化是什么？**

**正则化即Regularization，是对y=wx+b（分类任务中的判别函数，回归问题中的拟合函数）中参数w的约束**，通常按照惯例是不管b的，但如果在实际应用时在正则化函数里将b也加上的话，其实影响也不大，预测效果基本没差别。wiki中给出了对正则化的很好的解释：

1. 理论依据是试图将[奥卡姆剃刀原则](https://zh.wikipedia.org/wiki/%E5%A5%A5%E5%8D%A1%E5%A7%86%E5%89%83%E5%88%80)加入到求解过程
2. 从性质上看，正则化是对模型的复杂性施加惩罚
3. 从贝叶斯的观点来看，不同的正则化实现对应着对模型参数施加不同的先验分布
4. 从作用上看，可以将规则化作为一种技术来提高学习模型的泛化能力，即提高模型对新数据的预测能力

**2）参数越稀疏代表模型越简单吗？是否还有其它好处？**

是的。因为一个模型中真正重要的参数可能并不多，如果考虑所有的参数起作用，那么可以对训练数据可以预测的很好，但是对测试数据就可能较差。另一个好处是参数变少可以使整个模型获得更好的可解释性，而且还可以用做特征选择。

**3）参数值越小代表模型越简单吗？**

是的。因为越复杂的模型，越是会尝试对所有的样本进行拟合，甚至包括一些异常样本点，这就容易造成在较小的区间里预测值产生较大的波动，这种较大的波动也反映了在这个区间里的导数很大，而只有较大的参数值才能产生较大的导数，因此复杂的模型，其参数值会比较大。相反地，参数小，导数小，对于较大波动的样本特征具有鲁棒性。

我们可根据以上几个问题的答案，下面分析3种常见范数，即L0、L1、L2，对参数的约束，以及进而3种范数对模型复杂度和其它方面的影响。

##### 1.1 L0范数

L0是向量中非0的元素的个数。如果我们用L0范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0。换句话说，让参数W是稀疏的。

但不幸的是，L0范数的最优化问题是一个NP hard问题，而且理论上有证明，L1范数是L0范数的最优凸近似，因此通常使用L1范数来代替。

##### 1.2 L1范数


L1范数指向量中各个元素绝对值之和，也有个美称叫“稀疏规则算子”（Lasso regularization）。表示形式如下，

<center><img src="https://latex.codecogs.com/gif.latex?J(f)=\left\|\theta_f\right\|_1=\sum_{i=0}^{M}|\theta_{i}|"/></center>

L1正则化之所以可以防止过拟合，是因为L1范数就是各个参数的绝对值相加得到的，我们前面讨论了，参数值大小和模型复杂度是成正比的。因此复杂的模型，其L1范数就大，最终导致损失函数就大，说明这个模型就不够好。当特征参数为1维时，L1函数曲线如图X所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/L1.png"/>
</center>
<center>图X L1范数函数曲线（1维） </center>

从图X可以看到，


##### 1.3 L2范数


L2范数是向量中各个元素的平方和，也叫“岭回归”（Ridge Regression）。

与L1范数不一样的是，它不会是每个元素为0，而只是接近于0。越小的参数说明模型越简单，越简单的模型越不容易产生过拟合现象。

当特征参数为1维时，L2函数曲线如图X所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/L2.png"/>
</center>
<center>图X L2范数函数曲线（1维） </center>

从图X可以看到，相比与L1范数，L2处处可导。

##### 1.4 L1和L2的形象比较

接下来比较L1范数和L2范数分别对目标函数的影响，考虑目标函数的原始形式，

<center>
<img src="https://latex.codecogs.com/gif.latex?min\text{&space;}\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))&space;\\&space;s.t.\text{&space;}g(f)\leq0,\text{&space;}g(f)=J(f)-\alpha"/>
</center>

其中α为约束的常数，N是范数，上面这个式子很直观，表示在g(f)<=0的约束下，使L(y,f(x))最小，其中对于L1范数有，

<center><img src="https://latex.codecogs.com/gif.latex?J(f)=\left\|\theta_f\right\|_1=\sum_{i=0}^{M}|\theta_{i}|"/></center>

，对于L2范数有

<center><img src="https://latex.codecogs.com/gif.latex?J(f)=\left\|\theta_f\right\|_2=\sum_{i=0}^{M}\theta_{i}^2"/></center>

当 L(y,f(x)) 在无约束的条件下取最优解 f 本身就满足g(f)<=0，相当于没有约束，这种情况不考虑，因为此时与采用何种约束无关；**现在考虑另外一种更普遍的情况，f(θ)在无约束条件下最优解θ不满足g(θ)<=0，此时L1和L2应用于目标函数最优解求解的示意图可用图X表示，为了方便可视化，这里考虑特征参数为2维的情况**，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/L1&L2.png"/>
</center>
<center>图X L1范数和L2范数对模型参数的约束示意图 </center>

如图X所示，L(y,f(x)) 在无约束条件下最优解 f（指的是y=wx+b中特征参数x） 不满足 g(f)<=0，即绿色圆圆心在蓝色区域外，这种情况下，L(y,f(x)) 在 g(f)<=0 约束下的最优解必然在两者的边界相交处。基于以上知识，我们从图中可以看到对于同一个L(y,f(x))（绿色圆），通过L1范数约束（左子图蓝色正方形）和L2范数约束（右子图蓝色圆形）得到的最优解的位置有明显差别，L1范数约束得到的最优解在 θ1 轴上，而L2范数约束得到的最优解不在坐标轴上。

**在上述特征维度为2维的情况：** 对于L1范数只有无约束情况下最优解（绿色圆心）在两个平行红色虚线内时，在有约束情况下最优解（绿色圆和蓝色正方形交界处）才会在边（即非任意坐标轴）上，而绝大多数情况在顶点（即坐标轴）上；对于L2范数只有无约束情况下最优解（绿色圆心）本身就在坐标轴上时，在有约束情况下最优解才会在顶点（即坐标轴）上，而绝大多数情况在边（即非任意坐标轴）上。

**如果特征维度扩展到3维：** L(y,f(x))梯度可以简单看成是一个球体，而L1范数是一个八面体，两者的交界处也是很大概率在顶点上，意味着最优解的3个维度里有2个维度值是0；L2范数也是一个球，约束下最优解很大概率在非任意坐标轴上，故加L2范数的最优解通常每个维度都有值。

**虽然上述结论是在 L(y,f(x)) 梯度在各个维度是均匀的情况下得到的，但实际情况确实是加L1范数的最优解大多数维度值是0，加L2范数的最优解没有这个特性。**

##### 1.5 目标函数和KKT条件

在上一节即1.4的分析中提到，本文的目标函数的原始形式如下所示，

<center>
<img src="https://latex.codecogs.com/gif.latex?min\text{&space;}\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))&space;\\&space;s.t.\text{&space;}g(f)\leq0,\text{&space;}g(f)=J(f)-\alpha"/>
</center>

这个目标函数有一个不等式约束，它是怎么转换成本文开头图1所示方便求解的形式的呢？这就用到了下面的知识。在优化理论中，KKT（Karush-Kuhn-Tucher）条件是非线性规划（nonlinear programming)最佳解的必要条件。以下是KKT条件的定义，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/KKT.png"/>
</center>

下面根据KKT条件，具体分析本文的目标函数，L1范数和L2范数满足KKT条件的1-4条件，如下所示，假设 f0 是目标函数局部极小值对应的判别函数或回归函数，f也可以理解成函数的特征参数θ，

1. 本身范数的定义是**g(f0)<=0（满足条件4）**；
2. 由于最优解在范数内部时，范数等同于无约束，此时λ=0；最优解在范数的边界时理所当然范数g(f)=0。结合以上，故满足条件**λg(f0)=0（满足条件3）**；
3. 由于最优解是最值，所以**新目标函数导数为0（满足条件1）**；
4. 导数为0意味着满足L'(y,f(x))+λ·g'(f)=0，又由于f'(x,0)与g'(f)在最小值处符号相反，故**λ>=0（满足条件2）**。

那么目标函数可以通过拉格朗日乘子法改造成新的目标函数：

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/object function 1.png"/>
</center>

即本文最开始列出的目标函数形式，回到最初的起点。

### 二、贝叶斯学派眼中的L1范数和L2范数

假设有m个样本，频率学派上来就是用极大似然估计去估计最优参数，如下所示

<cetner><img src="https://latex.codecogs.com/gif.latex?P(S|\theta)=\prod_{i=1}^{m}p(y_i|x_i;\theta)=\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta)"/></center>

上式就是最大似然函数，这种方法求出θ有，

<center><img src="https://latex.codecogs.com/gif.latex?\theta_{MLE}=\arg\max_{\theta}&space;\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta)"  /></center>

贝叶斯学派觉得频率学派做法不妥，他们认为以上做法没有考虑θ的先验分布，也相当于P(θ)=Constant，θ的先验概率是一个常数；然而事实上，如果对P(θ)加上先验知识，式子应该变成下面的模样，

<cetner><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}P(\theta|S)&=\frac{P(S|\theta)P(\theta)}{P(S)}\\&space;&=\frac{(\prod_{i=1}^{m}p(y_i|x_i;\theta))P(\theta)}{\text{Consts}}\\&space;&=\frac{(\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta))P(\theta)}{\text{Consts}}&space;\end{aligned}" /></center>

最大后验概率求出θ为

<center><img src="https://latex.codecogs.com/gif.latex?\theta_{MLE}=\arg\max_{\theta}&space;(\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta))P(\theta)"  /></center>

比极大似然估计求的θ就多乘了一项P(θ)。以θ为自变量的函数L(θ)，

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(\theta)&=(\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta))P(\theta)&space;\\&space;ln(L(\theta))&=\sum_{i=1}^{m}ln[p(\epsilon_i=y_i-x_i^T\theta)])&plus;ln[P(\theta)]&space;\\&space;&=\sum_{i=1}^{m}ln[p(\epsilon_i=y_i-x_i^T\theta)])&plus;\sum_{i=1}^{n}ln[p(\theta_i)]&space;\end{aligned}" /></center>

对特征参数向量θ的每一维，假设服从拉普拉斯分布（laplace distribution）或假设服从高斯分布（Gaussian distribution），上述表达式变成下面的样子，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/laplace and gaussian.gif"/>
</center>

上式接近成目标函数的形式了。**L1范数相当于对特征参数θ每维假设服从拉普拉斯分布，L2范数相当于对特征参数θ每维假设服从高斯分布**。拉普拉斯分布和高斯分布如下图X所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/laplace & gaussian curve.png"/>
</center>

可以看到XX，然后我们会发现调整拉普拉斯分布的β值或高斯分布的σ值，相当于是在调整正则化系数。为了获取更具体的目标函数，我们对**y假设服从高斯分布**，则有

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/equation-lasso r & ridge r.gif"/>
</center>

可以看到，第一个式子就是OLS（最小二乘法）的Lasso Regression，第二个式子就是OLS的Ridge Regression。除此之外，当**y假设服从伯努利分布假设**，相当于加正则化的逻辑回归；同理，当**y假设服从拉普拉斯分布假设**，相当于加正则化的最小一乘法。

### 三、小结

用表格小结比较直观一些，

| 范数 | 表达式 | 先验知识 | 作用 | 
| :-----------:| :----------: | :----------: |:----------: |
| L1 | <img src="https://kangcai.github.io/img/in-post/post-ml/L1 latex.gif"/> | 拉普拉斯分布 | 1.产生一个稀疏模型，因此可以用于特征选择 <br> 2. 一定程度上可以防止过拟合（overfitting）|
| L2 | <img src="https://kangcai.github.io/img/in-post/post-ml/L2 latex.gif"/> | 高斯分布 | 主要作用是防止模型过拟合

1. [wiki: Regularization (mathematics)](https://en.wikipedia.org/wiki/Regularization_(mathematics))
2. [cnblogs: 机器学习之正则化](https://www.cnblogs.com/jianxinzhou/p/4083921.html])
3. [cnblogs: 机器学习中的范数规则化之（一）L0、L1与L2范数](https://www.cnblogs.com/weizc/p/5778678.html)
4. [jianshu: L0、L1、L2范数在机器学习中的应用](https://www.jianshu.com/p/4bad38fe07e6)
5. [jianshu: https://www.jianshu.com/p/c9bb6f89cfcc](https://www.jianshu.com/p/c9bb6f89cfcc)