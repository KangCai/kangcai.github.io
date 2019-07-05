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

跟很多机器学习算法一样，XXX 迭代

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

**完整代码见** [https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/code/svm.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/code/svm.py)

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

1. [wiki: 支持向量机](https://zh.wikipedia.org/wiki/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)
2. [cnblogs: 支持向量机（SVM）中的 SMO算法](https://www.cnblogs.com/bentuwuying/p/6444516.html)
3. [csdn: SVM（五）线性不可分之核函数](https://blog.csdn.net/The_lastest/article/details/78569217)