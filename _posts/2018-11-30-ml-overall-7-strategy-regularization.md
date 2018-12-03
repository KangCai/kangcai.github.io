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
<center>图1 机器学习模型的目标函数形式</center>

如图1所示，目标函数包含了表征经验风险的**代价函数**和表征结构风险的**正则化项**，上篇文章已经介绍了正则化系数和损失函数：正则化系数是设置代价函数和正则化项约束权重的常量，代价函数是样本集的损失函数之和。**下文将主要介绍正则化，正则化是对 y=wx+b 中参数 w（b也可以，但没必要）的约束**。


### 一、正则化

介绍正则化，首先回答下面三个问题：

**1) 机器学习中正则化是什么？**

**正则化即Regularization，是对 y=wx+b（分类任务中的判别函数，回归问题中的拟合函数）中参数 w 的约束，在目标函数中体现就是关于 w 的函数项**，通常按照惯例是不管b的，但如果在实际应用时将b也加上的话，其实影响也不大，预测效果基本没差别。wiki中给出了对正则化的很好的解释：

1. **理论依据是试图将[奥卡姆剃刀原则](https://zh.wikipedia.org/wiki/%E5%A5%A5%E5%8D%A1%E5%A7%86%E5%89%83%E5%88%80)加入到求解过程**
2. **从性质上看，正则化是对模型的复杂性施加惩罚**
3. **从贝叶斯的观点来看，不同的正则化实现对应着对模型参数施加不同的先验分布**
4. **从作用上看，可以将规则化作为一种技术来提高学习模型的泛化能力，即提高模型对新数据的预测能力**

后文将会对以上2-4点会进行详细的解释。

**2）参数越稀疏代表模型越简单、泛化能力越强吗？**

是的。因为特征中真正重要的维度可能并不多，反映到模型中就是真正重要的模型参数可能并不多，**保留尽可能少的重要的参数可以减弱非重要特征的波动带来的模型预测影响，使模型的泛化效果更好**。另一个好处是**参数变少可以使整个模型获得更好的可解释性，而且还可以用做特征选择**。

**3）参数值越小代表模型越简单、泛化能力越强吗？**

是的。因为越复杂的模型，越是会尝试对所有的样本进行拟合，这就容易造成在较小的区间里预测值产生较大的波动，对于一些异常样本点波动就更大，这种较大的波动也反映了在这个区间里的导数很大，而只有较大的参数值才能产生较大的导数，因此复杂的模型，其参数值会比较大。相反地，**参数小，则导数小，则对于样本特征的波动具有鲁棒性，那么模型泛化效果会更好**。

我们可带着以上3个问题的答案，来分析常见范数对参数的约束，内容包括L0、L1、L2的介绍以及它们对模型复杂度、拟合程度、特征向量稀疏性等方面的影响。

##### 1.1 L0范数

L0范数是向量中非0的元素的个数。如果我们用L0范数来规则化一个参数向量w的话，就是希望w的大部分元素都是0。换句话说，让参数w是稀疏的。

但不幸的是，**L0范数的最优化问题是一个NP hard问题；而且理论上有证明，L1范数是L0范数的最优凸近似，因此通常使用L1范数来代替**。

##### 1.2 L1范数


L1范数是向量中各个元素绝对值之和，也称为叫“稀疏规则算子”（Lasso regularization）。表示形式如下，

<center><img src="https://latex.codecogs.com/gif.latex?J(f)=\left\|\theta_f\right\|_1=\sum_{i=0}^{M}|\theta_{i}|"/></center>

L1正则化之所以可以防止过拟合，就是因为大的L1范数值表明模型不够好，具体分析如下：L1范数是各个参数的绝对值相加得到的，我们前面讨论了参数值大小和模型复杂度是成正比的，因此**复杂的模型，其L1范数就大，导致损失函数大，说明复杂模型就不够好，这样在一定程度防止过拟合现象**。另外，如上所说L1范数是L0范数的最优凸近似，它会使尽可能多的模型参数值为0，使参数具有稀疏性，详细分析在下文1.4节中给出。

##### 1.3 L2范数


L2范数是向量中各个元素的平方和，也叫“岭回归”（Ridge Regression）。表示形式如下，

<center><img src="https://latex.codecogs.com/gif.latex?J(f)=\left\|\theta_f\right\|_2=\sum_{i=0}^{M}\theta_{i}^2"/></center>

**与L1范数不一样的是，它不是使尽可能多的参数为0，而只是接近于0。越小的参数说明模型越简单，从而越不容易产生过拟合现象**。

##### 1.4 L1和L2的最优解稀疏性比较

接下来比较L1范数和L2范数分别对目标函数的影响，考虑目标函数的原始形式，

<center>
<img src="https://latex.codecogs.com/gif.latex?min\text{&space;}\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))&space;\\&space;s.t.\text{&space;}g(f)\leq0,\text{&space;}g(f)=J(f)-\alpha"/>
</center>

其中 α 为约束的常数，J(f) 是范数，f可以理解成 f(x) 中的特征参数 w。上面这个式子很直观，表示在 g(f)<=0 的约束下，使 L(y,f(x)) 最小，其中L1范数的 J(f) 如1.2节公式所示，和L2范数的 J(f) 如1.3节公式所示。下面我们就来详细分析目标函数在L1范数和L2范数两种不同约束下求得的最优解稀疏性的差别。

**1）无约束最优解本身就在约束内**

当 L(y, f(x)) 在无约束的条件下取最优解的 f 本身就满足 g(f)<=0 时，相当于没有 g(f)<=0 约束，这种情况不考虑，因为此时与采用何种约束无关。

**2）无约束最优解在约束外**

现在考虑另外一种更普遍的情况，即 L(y, f(x)) 在无约束条件下最优解的 f 不满足 g(θ)<=0，此时L1和L2应用于目标函数最优解求解的示意图可用图2表示，为了方便可视化，这里考虑特征参数为2维的情况，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/L1&L2.png"/>
</center>
<center>图2 L1范数和L2范数对模型参数的约束示意图 </center>

如图2所示，两个子图中，都是 **L(y,f(x)) 在无约束条件下最优解 f 不满足 g(f)<=0**，即绿色圆圆心在蓝色区域外，这种情况下，**L(y,f(x)) 在 g(f)<=0 约束下的最优解必然在两者的边界相交处**。基于以上知识，我们从图中可以看到对于同一个L(y,f(x))（绿色圆），**左子图中L1范数约束（图2左子图蓝色正方形）得到的最优解在 θ1 轴上，而右子图中L2范数约束（图2右子图蓝色圆形）得到的最优解不在坐标轴上**。

**3）特征维度为2时更普遍的结论**

如图2的左子图所示，对于L1范数，有且仅当无约束情况下最优解（绿色圆心）在两个平行红色虚线内时，在有约束情况下最优解（绿色圆和蓝色正方形交界处）才会在边（即非任意坐标轴）上；而绝大多数情况在顶点（即坐标轴）上；

如图2的右子图所示，对于L2范数，有且仅当无约束情况下最优解（绿色圆心）本身就在坐标轴上时，在有约束情况下最优解才会在顶点（即坐标轴）上；而绝大多数情况在非任意坐标轴上，与L1范数相反。

**4）如果特征维度扩展到3维**
 
 L(y,f(x)) 梯度是一个球体，而L1范数是一个八面体，两者的交界处也是很大概率在顶点上，意味着最优解的3个维度里有2个维度值是0；
 
 L2范数是一个球，约束下最优解很大概率在非任意坐标轴上，故加L2范数约束的最优解通常每个维度都有值。

**5）普遍结论**

**当维度扩展到更高，也有以上结论。值得一提的是，虽然上述结论是在 L(y,f(x)) 在各个维度梯度是均匀的情况下得到的，但通常情况即使各个维度梯度不是均匀的情况下，仍然是加L1范数约束的最优解大多数维度值是0，加L2范数约束的最优解每个维度值都不是0**。

##### 1.5 目标函数和KKT条件

在上一节1.4的分析中提到，本文的目标函数的原始形式如下所示，

<center>
<img src="https://latex.codecogs.com/gif.latex?min\text{&space;}\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))&space;\\&space;s.t.\text{&space;}g(f)\leq0,\text{&space;}g(f)=J(f)-\alpha"/>
</center>

这个目标函数带有一个不等式约束，那么它是怎么转换成本文开头图1所示方便求解的形式的呢？这就用到了下面的知识。**在优化理论中，KKT（Karush-Kuhn-Tucher）条件是非线性规划（nonlinear programming)最佳解的必要条件**。以下是KKT条件的定义，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/KKT.png"/>
</center>

根据KKT条件，下面具体分析本文的目标函数：L1范数和L2范数的不等式约束均满足KKT条件的1-4条件，如下所示，假设 f 是目标函数局部极小值对应的判别函数或回归函数，跟上文一样，f 也可以理解成函数的特征参数 θ，

1. 本身范数的定义是 **g(f)<=0，满足条件4**；
2. 由于最优解在范数内部时，范数等同于无约束，此时λ=0；最优解在范数的边界时，理所当然有范数g(f)=0。结合以上，故满足条件 **λg(f)=0，满足条件3**；
3. 由于最优解是最值，所以**新目标函数导数为0，满足条件1**；
4. 导数为0意味着满足 L'(y,f(x))+λ·g'(f)=0，又由于 L'(y,f(x)) 与 g'(f) 在最小值处符号相反，故 **λ>=0，满足条件2**。

那么目标函数可以通过拉格朗日乘子法改造成新的目标函数，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/object function 1.png"/>
</center>

即本文最开始列出的目标函数形式，回到最初的起点。

### 二、贝叶斯学派眼中的L1范数和L2范数

假设有m个样本，**频率学派上来就是用极大似然估计去估计最优参数**，如下所示

<center><img src="https://latex.codecogs.com/gif.latex?P(S|\theta)=\prod_{i=1}^{m}p(y_i|x_i;\theta)=\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta)"/></center>

上式就是最大似然函数，这种方法求出最优解 θ 有，

<center><img src="https://latex.codecogs.com/gif.latex?\theta_{MLE}=\arg\max_{\theta}&space;\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta)" /></center>

。贝叶斯学派觉得频率学派做法不妥，他们认为以上做法没有考虑 θ 的先验分布。在他们看来，这种做法相当于假设 P(θ)=Constant，即θ的先验概率是一个常数；事实上，如果加入先验知识，式子应该变成下面的模样，

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}P(\theta|S)&=\frac{P(S|\theta)P(\theta)}{P(S)}\\&space;&=\frac{(\prod_{i=1}^{m}p(y_i|x_i;\theta))P(\theta)}{\text{Consts}}\\&space;&=\frac{(\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta))P(\theta)}{\text{Consts}}&space;\end{aligned}" /></center>

，最大后验概率求得最优解 θ 为

<center><img src="https://latex.codecogs.com/gif.latex?\theta_{MLE}=\arg\max_{\theta}&space;(\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta))P(\theta)"  /></center>

，比极大似然估计求的θ就多乘了一项P(θ)。以θ为自变量的函数L(θ)，

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(\theta)&=(\prod_{i=1}^{m}p(\epsilon_i=y_i-x_i^T\theta))P(\theta)&space;\\&space;ln(L(\theta))&=\sum_{i=1}^{m}ln[p(\epsilon_i=y_i-x_i^T\theta)])&plus;ln[P(\theta)]&space;\\&space;&=\sum_{i=1}^{m}ln[p(\epsilon_i=y_i-x_i^T\theta)])&plus;\sum_{i=1}^{n}ln[p(\theta_i)]&space;\end{aligned}" /></center>

。对于特征参数向量θ的每一维，假设服从拉普拉斯分布（laplace distribution）或高斯分布（Gaussian distribution），上述表达式变成下面的样子，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/laplace and gaussian.gif"/>
</center>

，十分接近成目标函数的形式了。所以可以看到，**L1范数相当于假设特征参数 θ 每维服从拉普拉斯分布，L2范数相当于假设特征参数 θ 每维服从高斯分布**。拉普拉斯分布和高斯分布如下图3所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/laplace & gaussian curve.png"/>
</center>
<center>图3 拉普拉斯分布和高斯分布</center>

从图3可以看到**拉普拉斯分布靠近0处更加“陡峭”，意味着更倾向0值，即令特征向量稀疏**；而**高斯分布靠近0处相对比较“平缓”，但在远离0处相对拉普拉斯分布值更小，意味着倾向于小值，但不一定要求到0，即令特征向量θ各维度值小**。除此之外，**我们会发现调整拉普拉斯分布的 β 值或高斯分布的 σ 值，相当于是在调整正则化系数**。接下来，为了获取更具体的目标函数，我们**假设y服从高斯分布（注意，这是假设样本的分布，与假设特征参数的分布区分开）**，则有

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/equation-lasso r & ridge r.gif"/>
</center>

可以看到，**第一个式子就是OLS（最小二乘法）的Lasso Regression，第二个式子就是OLS的Ridge Regression**。除此之外，**假设 y 服从伯努利分布假设，相当于加正则化的逻辑回归**；同理，**假设 y 服从拉普拉斯分布假设，相当于加正则化的最小一乘法**。

### 三、小结

用表格小结比较直观一些，

| 范数 | 表达式 | 参数先验知识 | 作用 | 
| :-----------:| :----------: | :----------: |:----------: |
| L1 | <img src="https://kangcai.github.io/img/in-post/post-ml/L1 latex.gif"/> | 拉普拉斯分布 | 1）产生稀疏模型，可以用于特征选择 <br> 2）一定程度上可以防止过拟合（overfitting）|
| L2 | <img src="https://kangcai.github.io/img/in-post/post-ml/L2 latex.gif"/> | 高斯分布 | 主要作用是防止模型过拟合

三要素之策略，为模型的好与坏定下了评判的标准，正则化作为评判标准的一部分，主要起到防止模拟过拟合或产生稀疏模型的作用。在实际应用中，正则化使用得相当广泛和频繁，在一定程度上是机器学习模型不可或缺的一部分。

1. [wiki: Regularization (mathematics)](https://en.wikipedia.org/wiki/Regularization_(mathematics))
2. [cnblogs: 机器学习之正则化](https://www.cnblogs.com/jianxinzhou/p/4083921.html])
3. [cnblogs: 机器学习中的范数规则化之（一）L0、L1与L2范数](https://www.cnblogs.com/weizc/p/5778678.html)
4. [jianshu: L0、L1、L2范数在机器学习中的应用](https://www.jianshu.com/p/4bad38fe07e6)
5. [jianshu: https://www.jianshu.com/p/c9bb6f89cfcc](https://www.jianshu.com/p/c9bb6f89cfcc)