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

### 一、概念

### 二、算法步骤

根据第一节，SVM的优化目标和约束条件如下，

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;max\&space;W(\alpha)&=\sum_{i=1}^{n}\alpha-\frac{1}{2}\sum_{i,j=1}^{n}y_iy_ja_ia_j(K(x_i,x_j))\\&space;s.t.\sum_{i=1}^{n}y_ia_i&=0&space;\\&space;0&space;\leq&space;a_i&space;\leq&space;C&(i=1,2...n)&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;max\&space;W(\alpha)&=\sum_{i=1}^{n}\alpha-\frac{1}{2}\sum_{i,j=1}^{n}y_iy_ja_ia_j(K(x_i,x_j))\\&space;s.t.\sum_{i=1}^{n}y_ia_i&=0&space;\\&space;0&space;\leq&space;a_i&space;\leq&space;C&(i=1,2...n)&space;\end{aligned}" title="\begin{aligned} max\ W(\alpha)&=\sum_{i=1}^{n}\alpha-\frac{1}{2}\sum_{i,j=1}^{n}y_iy_ja_ia_j(K(x_i,x_j))\\ s.t.\sum_{i=1}^{n}y_ia_i&=0 \\ 0 \leq a_i \leq C&(i=1,2...n) \end{aligned}" /></a>

关于这种二次规划问题求解的方法有很多，这里具体采用的是 SMO （Sequential minimal optimization）优化算法。SMO 算法是一种启发式算法，基本思路是：如果所有变量的解都满足此最优化问题的 KKT 条件，那么这么最优化问题的解就得到了。具体进行计算时，它采用分解的思想，每次只优化两个点 {i, j} 的工作集，算法步骤如下，

1.根据当前参数 alpha 计算当前分割超平面的 w 和 b

```buildoutcfg
w = np.dot(X_train.T, np.multiply(alpha, Y_train))
b = np.mean(Y_train - np.dot(w.T, X_train.T))
```

2.分别计算关于样本 i 和 j 在本轮中的预测误差，留作后用

```buildoutcfg
def predict(self, X):
    return np.sign(np.dot(self.w.T, X.T) + self.b).astype(int)
    
E_i = predict(x_i) - y_i
E_j = predict(x_j) - y_j
```

3.计算关于 x_i 和 x_j 的核函数值，留作后用

```buildoutcfg
// 比如多项式核函数
def kernel_func(x1, x2):
    return np.dot(x1, x2.T) ** 2

k_ij = kernel_func(x_i, x_i) + kernel_func(x_j, x_j) - 2 * kernel_func(x_i, x_j)
```

4.计算更新后 alpha\[j\] 的边界值，留做后用

```buildoutcfg
L, H = self._cal_L_H(self.C, a_j, a_i, y_j, y_i)
```

5.根据上述 预测误差、核函数值、alpha\[j\] 边界值，更新 alpha\[j\]

```buildoutcfg
alpha[j] = a_j + (y_j * (E_i - E_j) * 1.0) / k_ij
alpha[j] = min(H, max(L, alpha[j]))
```

6.根据 alpha\[j\] 计算 alpha\[i\] 

```buildoutcfg
alpha[i] = a_i + y_i * y_j * (a_j - alpha[j])
```

所以，最后训练复杂度是 XXX。

**完整代码见** [https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/support_vector_machine.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/support_vector_machine.py)

### 三、表现效果

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/svm_1.png"/>
</center>

除此之外，如果借助 scikit-learn 实现，老样子几行搞定，

```buildoutcfg
from sklearn import svm
svm_model = svm.SVC()
lr.fit(X, Y)
result_predict = lr.predict(X')
```

### 四、Q&A

**1.SVM 的实现除了 SMO 还能用哪些优化算法，为什么通常用 SMO？**

SVM 将原始问题转化为对偶问题后，问题就是一个凸二次规划问题，理论上任何一个解决凸二次规划问题都可以解决该问题。然而一般的方法通常会很慢，SMO 基于问题本身的特性（KKT条件约束），对这个特殊的二次规划问题的求解过程进行了优化。具体来说就是，如果优化任务中所有变量的解都满足此最优化问题的KKT条件，那么这么最优化问题的解就得到了，所以在固定其他参数以后，这就是一个单变量二次规划问题，这样就有直接可得的解析解（analytical solution，或称闭式解，closed-form solution），求解效率高。

**2.如何加速计算**

所谓加速计算，就是找到一个使目标函数增大最快的方法，一个重要的优化点在于每次迭代的两个样本的选取方法。对于每个样本都要遍历到，比如说需要遍历的样本是 j，配合该样本进行迭代的样本是 i，j 按顺序遍历即可，而 i 的选择会直接影响算法的收敛速度。每次直接选使目标函数增大最大的样本 i 是不可取的，因为着需要计算所有样本，效率更低；一种可取的方法是选取与样本 j 差别很大的样本，因为直观来说，更新两个差别很大的变量比起相似的变量会带给目标函数更大的变化，具体可以通过 Hinge 函数 E_i = max(y_i * f(x_i) - 1, 0) 来找到使 |E_i - E_j| 最大的样本 i。


**参考文献**

1. [wiki: 支持向量机](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)
2. [cnblogs: 支持向量机（SVM）中的 SMO算法](https://www.cnblogs.com/bentuwuying/p/6444516.html)
3. [csdn: SVM（五）线性不可分之核函数](https://blog.csdn.net/The_lastest/article/details/78569217)