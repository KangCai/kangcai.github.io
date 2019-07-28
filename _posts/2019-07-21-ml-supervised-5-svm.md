---
layout: post
title: "机器学习 · 监督学习篇 V"
subtitle: "支持向量机"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

> 支持向量机是一种经典的二分类判别模型，基本定义是在特征空间上的间隔最大的线性分类器。1995年，Vapnik 和 Cortes 发表软间隔支持向量机（SVM算法），开启了随后的机器学习领域 NN 和 SVM 两大社区持续十多年的竞争历程

> 文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

### 一、概念

支持向量机是一种经典的二分类判别模型，基本定义是在特征空间上的**间隔最大的线性分类器**，它具有以下几个特性：具备独有的优化目标，即判别超平面与特征的**间隔最大化**；可以**利用核技巧，成为非线性分类器**；通过**将优化问题转化为其对偶问题**，支持向量机的学习算法等价于求解凸二次规划的最优化算法。

间隔最大及相应的约束最优化问题将在后文叙述，这里先从间隔的基本概念说起。高中我们学过点到直线的距离，比如对于直线函数式

<center><img src="https://latex.codecogs.com/gif.latex?2x&plus;y-1=0" /></center>

点 (x， y) 到该直线的距离为

<center><img src="https://latex.codecogs.com/gif.latex?d=\frac{2x&plus;y-1}{\sqrt{2^2&plus;1^2}}" /></center>

，如下图所示

<center><img src="https://kangcai.github.io/img/in-post/post-ml/svm_dist.png" width=50%/></center>

以上是二维平面的情况，从**二维空间拓展到多维空间**下，点到超平面的计算也是类似的，假设超平面函数表达式为

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&wx&plus;b=0&space;\\&space;\Rightarrow\&space;&\theta_1x_1&plus;\theta_2x_2&plus;...&plus;\theta_Mx_M&plus;b=0&space;\end{aligned}"/></center>

，则点

<center><img src="https://latex.codecogs.com/gif.latex?x^{'}=(x_1，&space;x_2，...x_M)"/></center>

到超平面的距离可表示为

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&d=\frac{wx&plus;b}{\left&space;\|&space;w&space;\right&space;\|}&space;\\&space;&=\frac{\theta_1x_1&plus;\theta_2x_2&plus;...&plus;\theta_Mx_M&plus;b}{\sqrt{\theta_1^2&plus;\theta_2^2&plus;...&plus;\theta_M^2}}&space;\end{aligned}"/></center>

这个距离是正负号分别表示点在超平面的两侧，所以如果要考虑样本标签的正负属性（-1, 1）与超平面两个侧面关系对应的话，非负距离可表示为

<center><img src="https://latex.codecogs.com/gif.latex?\gamma=yd=\frac{y(wx&plus;b)}{\left&space;\|&space;w&space;\right&space;\|}"></center>


**SVM 的优化目标是使样本到超平面的间隔最大，可以拆解为两层意思，一层是所有样本到超平面的距离都大于某个距离，另一层是使该距离最大化**，可以用以下公式表示，

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&\g\max_{w,b}\gamma&space;\\&space;&&space;s.t.\&space;y_i(w\cdot&space;x_i&plus;b)/\left&space;\|&space;w&space;\right&space;\|\geq\gamma&space;\end{aligned}&space;\Rightarrow&space;\begin{aligned}&space;&\g\max_{w,b}\hat{\gamma}&space;/&space;\left&space;\|&space;w&space;\right&space;\|\\&space;&&space;s.t.\&space;y_i(w\cdot&space;x_i&plus;b)\geq\hat{\gamma}&space;\end{aligned}&space;\Rightarrow&space;\begin{aligned}&space;&\g\min_{w,b}\left&space;\|&space;w&space;\right&space;\|\\&space;&&space;s.t.\&space;y_i(w\cdot&space;x_i&plus;b)\geq&space;1&space;\end{aligned}"/>
</center>

**最小化一个非负数与最小化其常数倍的平方数等价**，于是就得到了下面的线性可分支持向量机学习的最优化问题

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\min&space;_{\gamma，&space;w，&space;b}&space;&&space;\frac{1}{2}\|w\|^{2}&space;\\&space;\text&space;{&space;s.t.&space;}&space;&&space;y^{(i)}\left(w^{T}&space;x^{(i)}&plus;b\right)&space;\geq&space;1，&space;\quad&space;i=1，&space;\ldots，&space;m&space;\end{aligned}"/></center>

为了求解原始最优化问题，**应用拉格朗日对偶性，通过求解对偶问题（dual problem）得到原始问题（primal problem）的最优解**。这样做有两个优点，一是**对偶问题往往更容易求解**；二是能**自然引入核函数，进而将线性可分支持向量机推广到非线性分类问题**。具体操作是先定义原始问题的拉格朗日函数，

<center>
<img src="https://latex.codecogs.com/gif.latex?L(w,b,\alpha)=\frac{1}{2}\left&space;\|&space;w&space;\right&space;\|^2-\sum_{i=1}^{N}\alpha_iy_i(w\cdot&space;x_i&plus;b)&plus;\sum_{i=1}^{N}\alpha_i" />
</center>

**分别求 L 关于 w 和 b 的偏导数，并令其为0，将得到的两个等式一起带入上述拉格朗日函数（这两步省略），就将原始问题转化为对偶问题**，如下所示，

<center><img src="https://latex.codecogs.com/gif.latex?\begin{array}{cl}{\max&space;_{\alpha}}&space;&&space;{W(\alpha)=\sum_{i=1}^{m}&space;\alpha_{i}-\frac{1}{2}&space;\sum_{i，&space;j=1}^{m}&space;y^{(i)}&space;y^{(j)}&space;\alpha_{i}&space;\alpha_{j}\left\langle&space;x^{(i)}，&space;x^{(j)}\right\rangle}&space;\\&space;{\text&space;{&space;s.t.&space;}}&space;&&space;{\alpha_{i}&space;\geq&space;0，&space;\quad&space;i=1，&space;\ldots，&space;m}&space;\\&space;{}&space;&&space;{\sum_{i=1}^{m}&space;\alpha_{i}&space;y^{(i)}=0}\end{array}" /></center>

以上是针对硬间隔数据的 SVM 算法公式，所谓**硬间隔，就是说数据样本是可以实现线性可分，即存在分隔超平面完全将正负样本分开。然后，现实中大多数情况下 SVM 要解决的是软间隔问题，即数据样本不是实际的线性可分，而是近似线性可分**。

线性不可分意味着某些样本点(xi，yi)不能满足函数间隔大于等于1的约束条件，**为了解决软间隔问题，SVM 对每个样本点引入一个松弛变量**，降低实际的“函数间隔”。也就是松弛变量加上理论函数间隔大于等于1。所以原优化问题可以写成

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\min&space;_{\gamma，&space;w，&space;b}&space;&&space;\frac{1}{2}\|w\|^{2}&plus;C&space;\sum_{i=1}^{m}&space;\xi_{i}&space;\\&space;\text&space;{&space;s.t.&space;}&space;&&space;y^{(i)}\left(w^{T}&space;x^{(i)}&plus;b\right)&space;\geq&space;1-\xi_{i}，&space;\quad&space;i=1，&space;\ldots，&space;m&space;\\&space;&&space;\xi_{i}&space;\geq&space;0，&space;\quad&space;i=1，&space;\ldots，&space;m&space;\end{aligned}" /></center>

**目标函数中的 C 是惩罚参数，C>0，C 的值由我们决定，C 值大对误分类的惩罚增大，C 值小对误分类的惩罚小**。然后相应的，对偶问题就变成了

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\max&space;_{\alpha}&space;&&space;W(\alpha)=\sum_{i=1}^{m}&space;\alpha_{i}-\frac{1}{2}&space;\sum_{i，&space;j=1}^{m}&space;y^{(i)}&space;y^{(j)}&space;\alpha_{i}&space;\alpha_{j}\left\langle&space;x^{(i)}，&space;x^{(j)}\right\rangle&space;\\&space;\text&space;{&space;s.t.&space;}&space;&&space;0&space;\leq&space;\alpha_{i}&space;\leq&space;C，&space;\quad&space;i=1，&space;\ldots，&space;m&space;\\&space;&&space;\sum_{i=1}^{m}&space;\alpha_{i}&space;y^{(i)}=0&space;\end{aligned}" /></center>

这样，SVM 的算法求解问题就变成了一个凸二次规划问题。

### 二、算法步骤

根据第一节，SVM的优化目标和约束条件如下，

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;max\&space;W(\alpha)&=\sum_{i=1}^{n}\alpha-\frac{1}{2}\sum_{i，j=1}^{n}y_iy_ja_ia_j(K(x_i，x_j))\\&space;s.t.\sum_{i=1}^{n}y_ia_i&=0&space;\\&space;0&space;\leq&space;a_i&space;\leq&space;C&(i=1，2...n)&space;\end{aligned}" title="\begin{aligned} max\ W(\alpha)&=\sum_{i=1}^{n}\alpha-\frac{1}{2}\sum_{i，j=1}^{n}y_iy_ja_ia_j(K(x_i，x_j))\\ s.t.\sum_{i=1}^{n}y_ia_i&=0 \\ 0 \leq a_i \leq C&(i=1，2...n) \end{aligned}" /></center>

关于这种凸二次规划问题求解的方法有很多，但其它算法的时间复杂度都很高，这里具体采用的是**复杂度最低的 SMO （Sequential minimal optimization）优化算法**。SMO 算法是一种启发式算法，**基本思路是：如果所有变量的解都满足此最优化问题的 KKT 条件，那么这么最优化问题的解就得到了**。具体进行计算时，它采用分解的思想，**每次只优化两个点 {i， j} 的工作集**，算法步骤如下，

1.根据当前参数 alpha 计算当前分割超平面的 w 和 b

```buildoutcfg
w = np.dot(X_train.T， np.multiply(alpha， Y_train))
b = np.mean(Y_train - np.dot(w.T， X_train.T))
```

2.分别计算关于样本 i 和 j 在本轮中的预测误差，留作后用

```buildoutcfg
def predict(self， X):
    return np.sign(np.dot(self.w.T， X.T) + self.b).astype(int)
    
E_i = predict(x_i) - y_i
E_j = predict(x_j) - y_j
```

3.计算关于 x_i 和 x_j 的核函数值，留作后用

```buildoutcfg
// 比如多项式核函数
def kernel_func(x1， x2):
    return np.dot(x1， x2.T) ** 2

k_ij = kernel_func(x_i， x_i) + kernel_func(x_j， x_j) - 2 * kernel_func(x_i， x_j)
```

4.计算更新后 alpha\[j\] 的边界值，留做后用

```buildoutcfg
L， H = self._cal_L_H(self.C， a_j， a_i， y_j， y_i)
```

5.根据上述 预测误差、核函数值、alpha\[j\] 边界值，更新 alpha\[j\]

```buildoutcfg
alpha[j] = a_j + (y_j * (E_i - E_j) * 1.0) / k_ij
alpha[j] = min(H， max(L， alpha[j]))
```

6.根据 alpha\[j\] 计算 alpha\[i\] 

```buildoutcfg
alpha[i] = a_i + y_i * y_j * (a_j - alpha[j])
```

整个算法训练过程的时间复杂度平均是 O(n^2)，最坏是 O(n^3)。

**完整代码见** [https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/support_vector_machine.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/support_vector_machine.py)

### 三、表现效果

下图是用上一节算法代码运行一个示例的结果，

<center><img src="https://kangcai.github.io/img/in-post/post-ml/svm_1.png"/ width=60%></center>

图中每个点通过训练后对应两个数字，用逗号隔开，前一个数字表示该样本对应的 alpha 值，表示对分类超平面的贡献度；后一个数字表示点到分类超平面的几何间隔。

除此之外，如果借助 scikit-learn 实现，老样子几行搞定，

```buildoutcfg
from sklearn import svm
svm_model = svm.SVC()
lr.fit(X， Y)
result_predict = lr.predict(X')
```

### 四、Q&A

**1.SVM 的实现除了 SMO 还能用哪些优化算法，为什么通常用 SMO？**

SVM 将原始问题转化为对偶问题后，问题就是一个凸二次规划问题，理论上任何一个解决凸二次规划问题都可以解决该问题。然而一般的方法通常会很慢，SMO 基于问题本身的特性（KKT条件约束），对这个特殊的二次规划问题的求解过程进行了优化。具体来说就是，如果优化任务中所有变量的解都满足此最优化问题的KKT条件，那么这么最优化问题的解就得到了，所以在固定其他参数以后，这就是一个单变量二次规划问题，这样就有直接可得的解析解（analytical solution，或称闭式解，closed-form solution），求解效率高。

**2.如何加速计算**

所谓加速计算，就是找到一个使目标函数增大最快的方法，一个重要的优化点在于每次迭代的两个样本的选取方法。对于每个样本都要遍历到，比如说需要遍历的样本是 j，配合该样本进行迭代的样本是 i，j 按顺序遍历即可，而 i 的选择会直接影响算法的收敛速度。每次直接选使目标函数增大最大的样本 i 是不可取的，因为着需要计算所有样本，效率更低；一种可取的方法是选取与样本 j 差别很大的样本，因为直观来说，更新两个差别很大的变量比起相似的变量会带给目标函数更大的变化，具体可以通过 Hinge 函数 E_i = max(y_i * f(x_i) - 1， 0) 来找到使 \|E_i - E_j\| 最大的样本 i。


**参考文献**

1. [wiki: 支持向量机](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)
2. [cnblogs: 支持向量机（SVM）中的 SMO算法](https://www.cnblogs.com/bentuwuying/p/6444516.html)
3. [jianshu: SVM 由浅入深的尝试（四） 软间隔最大化的理解](https://www.jianshu.com/p/c4acd8c323ab)
4. [csdn: SVM（五）线性不可分之核函数](https://blog.csdn.net/The_lastest/article/details/78569217)