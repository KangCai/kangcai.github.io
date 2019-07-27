---
layout: post
title: "机器学习 · 监督学习篇 VI"
subtitle: "感知机"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

> 感知机（perception）是二类分类的线性判别模型，由美国心理学家于1957年提出。感知机是第一个从算法上完整描述的神经网络，是很多算法的鼻祖，比如支持向量机算法，神经网络与深度学习。

> 文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

### 一、概念

1957 美国心理学家 Rosenblatt 提出了感知机模型，作为一个二类分类的线性判别模型，它具有简单而易于实现的优点。支持向量机 于 1995 年才提出，感知机作为支持向量机的基础，算法思想与线性支持向量机有诸多相似之处：比如算法目标也是找到一个合适的分隔超平面，模型优化问题具有原始形式和对偶形式这两种形式，最初的模型只能解决线性数据分类问题。

与 SVM 类似，距离和算法公式（TODO）

<center><img src="https://latex.codecogs.com/gif.latex?L(w,&space;b)=-\sum_{i=1}^{F}&space;y_{i}\left(w&space;\cdot&space;x_{i}&plus;b\right)" /></center>


### 二、算法步骤 - 原始形式

（TODO）

<center><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\nabla_{w}&space;L(w,&space;b)&space;&=-\sum_{x_{i}&space;\in&space;M}&space;y_{i}&space;x_{i}&space;\\&space;\nabla_{b}&space;L(w,&space;b)&space;&=-\sum_{x_{i}&space;\in&space;M}&space;y_{i}&space;\end{aligned}&space;\Rightarrow&space;\begin{array}{l}{w&space;\leftarrow&space;w&plus;\eta&space;y_{i}&space;x_{i}}&space;\\&space;{b&space;\leftarrow&space;b&plus;\eta&space;y_{i}}\end{array}" /></center>

算法实现代码就很简单，循环迭代，直到分类完全没有错误才停止，

```buildoutcfg
// n_iter 是最多迭代次数，error_count_history 记录每次迭代的分类错误数
error_count_history = []
for _ in range(n_iter):
    error_count = 0
    for xi, yi in zip(X_train, Y_train):
        if yi * self.predict(xi) <= 0:
            self.w += self.eta * yi * xi
            self.b += self.eta * yi
            error_count += 1
    error_count_history.append(error_count)
    if error_count == 0:
        break
```

### 三、算法步骤 - 对偶形式

原始形式中，我们每次迭代对误分类点执行以下操作，

<center><img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}{w&space;\leftarrow&space;w&plus;\eta&space;y_{i}&space;x_{i}}&space;\\&space;{b&space;\leftarrow&space;b&plus;\eta&space;y_{i}}\end{array}" /></center>

就是逐步修改 w 和 b，假设迭代了 n 次，则 w, b 如果都是从零向量或者零开始迭代的话，那么学习到的 w 和 b 等价于

<center><img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}{w=\sum_{i=1}^{N}\alpha_iy_ix_i}&space;\\&space;{b=\sum_{i=1}^{N}\alpha_iy_i}&space;\end{array}" /></center>

其中 alpha 都是非负数，这样一来，我们的求解目标就从求超平面的参数 w 和 b 转化成了求每个样本的系数 alpha，具体操作是：每次分类错误一个样本，就对该样本的系数加 1 个单位的增量，即学习率。但这种方法有一个问题就是判断样本分类是否错误的复杂度比较高，因为每次都要根据每个样本的 alpha 来算 w，即判断公式变成了

<center><img src="https://latex.codecogs.com/gif.latex?y_i(\sum_{j=1}^{N}a_jy_jx_j\cdot&space;x_i&plus;b)\leq&space;0" /></center>

这样一来复杂度还挺高的，好在这个时候公式中的 x 和 y 都是不变量，我们可以将计算量最高的 x_j 与 x_i 的点乘结果提前存下来，用空间换时间，这个存结果的矩阵称之为格拉姆矩阵（Gram matrix），如下所示

```buildoutcfg
// Gram matrix
self.Gram_matrix = np.dot(X_train, X_train.T)
```

利用格拉姆矩阵，我们能快速地进行每次地迭代计算，

```buildoutcfg
// Iteration
i = 0
while i < n_samples:
    # Judge end of iteration
    wx = np.sum(np.dot(self.Gram_matrix[i, :] , self.alpha * Y_train))
    if Y_train[i] * (wx + self.b) <= 0:
        # a <- a + eta, b <- b + eta * y_i
        self.alpha[i] += self.eta
        self.b += self.eta * Y_train[i]
        i = 0
    else:
        i += 1
```
w 和 b感知机算法对偶形式为什么在实际运用中比原始形式快

### 四、表现效果

用随机生成的一些数据，运行上述原始形式和对偶形式的算法代码，得到的分类结果如下所示，

<center><img src="https://kangcai.github.io/img/in-post/post-ml/perception_3.png" width=45%/><img src="https://kangcai.github.io/img/in-post/post-ml/perception_1.png" width=45%/></center>

其中绿色和蓝色的点分别表示两类样本，



线性不可分的过程

<center><img src="https://kangcai.github.io/img/in-post/post-ml/perception_4.png" width=50%/></center>

1. [wiki: 感知器](https://zh.wikipedia.org/wiki/%E6%84%9F%E7%9F%A5%E5%99%A8)
2. [cnblogs: 感知机原理（Perceptron）](https://www.cnblogs.com/huangyc/p/9706575.html)