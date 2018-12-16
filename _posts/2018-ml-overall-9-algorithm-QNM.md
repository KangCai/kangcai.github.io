---
layout: post
title: "机器学习 · 总览篇 IX"
subtitle: "三要素之算法 - 梯度下降法及其变种"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

>

> 算法。文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)

牛顿法与拟牛顿法，DFP法，BFGS法，L-BFGS法
高斯-牛顿法（GD）、通常是针对非线性最小二乘问题，我们将在《机器学习·有监督学习篇》中的某一篇对解决最小二乘问题的各种算法进行详细的介绍。

### 一、牛顿法

##### 1.1 原始的牛顿法（Newton's method）

**用于求方程解**

牛顿法用于求解<img src="http://latex.codecogs.com/gif.latex?f(x)=0" />问题时，首先则是先设定初始值<img src="http://latex.codecogs.com/gif.latex?x_0" title="x_0" />，将问题转化为求 <img src="http://latex.codecogs.com/gif.latex?f'(x_0)=0" /> 这个方程的根，然后求得的根作为下一次迭代的 _x_ 。

**用于最优化问题**

当牛顿法用于机器学习的最优化问题时，目标是求解 <img src="http://latex.codecogs.com/gif.latex?L'(\theta)=0"/> 由于公式为 <img src="http://latex.codecogs.com/gif.latex?\theta^{t+1}\leftarrow \theta^t - \frac{L'(\theta)}{L''(\theta)} "/> 。

海森矩阵是函数 <img src="http://latex.codecogs.com/gif.latex?f"/> 的二阶偏导，

<center>
<img src="http://latex.codecogs.com/gif.latex?H_f=\begin{bmatrix}\frac{\partial^2f}{\partial&space;x_0^2}&\frac{\partial^2f}{\partial&space;x_0&space;\partial&space;x_1}&...&\frac{\partial^2f}{\partial&space;x_0&space;\partial&space;x_n}\\&space;\frac{\partial^2f}{\partial&space;x_1&space;\partial&space;x_0}&\frac{\partial^2f}{\partial&space;x_1^2}&...&\frac{\partial^2f}{\partial&space;x_1&space;\partial&space;x_n}\\\vdots&\vdots&\ddots&\vdots\\&space;\frac{\partial^2f}{\partial&space;x_n&space;\partial&space;x_0}&\frac{\partial^2f}{\partial&space;x_n&space;\partial&space;x_1}&...&\frac{\partial^2f}{\partial&space;x_n^2}&space;\end{bmatrix}"/>
</center>

**优点：二阶收敛，收敛速度快**

从几何上说，牛顿法就是用一个二次曲面去拟合你当前所处位置的局部曲面，而相比于梯度下降法，梯度下降法是用一个平面去拟合当前的局部曲面，通常情况下，二次曲面的拟合会比平面更好，所以牛顿法选择的下降路径会更符合真实的最优下降路径，路径找对了，下降速度就更快。

**缺点：难以计算**

牛顿法是一种迭代算法，每一步都需要求解目标函数的Hessian矩阵的逆矩阵，计算比较复杂甚至是无法计算，所以在机器学习中一般不会直接使用牛顿法。

##### 1.2 高斯-牛顿法（Gauss-Newton algorithm，GN）

是求解最小二乘问题的一个特例，

https://www.cnblogs.com/monoSLAM/p/5246665.html

GN 是对牛顿法的改进，解决高维牛顿法难解决的计算问题。它用到两个矩阵，雅可比矩阵（Jacobian matrix）。其中雅可比矩阵是函数 <img src="http://latex.codecogs.com/gif.latex?f"/> 的一阶偏导矩阵，

<center>
<img src="http://latex.codecogs.com/gif.latex?J_f=\begin{bmatrix}&space;\frac{\partial&space;f}{\partial&space;x_0}&\cdots&\frac{\partial&space;f}{\partial&space;x_n}\\&space;\vdots&\ddots&\vdots\\&space;\frac{\partial&space;f}{\partial&space;x_0}&\cdots&\frac{\partial&space;f}{\partial&space;x_n}&space;\end{bmatrix}" />
</center>

牛顿法公式可转换成，

<center>
<img src="http://latex.codecogs.com/gif.latex?X_{n&plus;1}=X_n-H_f(x_n)^{-1}\nabla&space;f(x_n)" />
</center>

由于 _x_ 是多维的，梯度向量在 _x_ 的第 _i_ 维上分量为

g

这样有两个好处，第一是线性，第二是只需要一阶微分。缺点是GN之反矩阵不存在时，而且在接近极值点时，Hessian矩阵变得非常小，可以类比成缩小了梯度下降法的学习率，。

##### 1.3 莱文贝格-马夸特算法（Levenberg–Marquardt algorithm，LM）

其实就只是在GN基础上修改了一点，GN中，迭代公式为，而在LM方法中，。从LM的公式中可以看到，<img src="http://latex.codecogs.com/gif.latex?\lambda" /> 大的时候这种算法会接近GD，小的时候会接近GN。在LM的实际应用中，为了保证能快速稳定下降，通常会动态调整<img src="http://latex.codecogs.com/gif.latex?\lambda" />：先使用较小的<img src="http://latex.codecogs.com/gif.latex?\lambda" />，使之更接近GN，可以快速下降；

### 二、拟牛顿法（Quasi-Newton Methods）

拟牛顿法是求解非线性优化问题最有效的方法之一，于20世纪50年代由美国Argonne国家实验室的物理学家W.C.Davidon所提出来。Davidon设计的这种算法在当时看来是非线性优化领域最具创造性的发明之一。不久R. Fletcher和M. J. D. Powell证实了这种新的算法远比其他方法快速和可靠，使得非线性优化这门学科在一夜之间突飞猛进。

拟牛顿法的本质思想是改善牛顿法每次需要求解复杂的Hessian矩阵的逆矩阵的缺陷，它使用正定矩阵来近似Hessian矩阵的逆，从而简化了运算的复杂度。

##### 1.1 DFP算法（Davidon-Fletcher-Powell algorithm）

##### 1.2 BFGS算法（Broyden–Fletcher–Goldfarb–Shanno algorithm）

##### 1.3 L-BFGS法（Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm）


### 三、拟牛顿法和梯度下降法的比较


！！！https://www.zhihu.com/question/46441403
https://www.cnblogs.com/8335IT/p/5809495.html 牛顿法和梯度下降法对比
http://blog.sina.com.cn/s/blog_1442877660102wru5.html


[csdn: 最全的机器学习中的优化算法介绍](https://blog.csdn.net/qsczse943062710/article/details/76763739)
[cnblog: 常见的几种最优化方法](http://www.cnblogs.com/maybe2030/p/4751804.html)
[wiki: Levenberg-Marquardt方法](https://zh.wikipedia.org/wiki/%E8%8E%B1%E6%96%87%E8%B4%9D%E6%A0%BC%EF%BC%8D%E9%A9%AC%E5%A4%B8%E7%89%B9%E6%96%B9%E6%B3%95)
[wiki: 海森矩阵](https://zh.wikipedia.org/wiki/%E6%B5%B7%E6%A3%AE%E7%9F%A9%E9%98%B5)
[wiki: 雅可比矩阵](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5)
[csdn: Levenberg-Marquardt算法浅谈](https://blog.csdn.net/liu14lang/article/details/53991897)

