---
layout: post
title: "机器学习·有监督学习篇 II"
subtitle: "最小二乘问题的求解"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习总览
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)

### 频

**牛顿法（Newton）**

（1）牛顿法用于求解<img src="http://latex.codecogs.com/gif.latex?f(x)=0" />问题时，首先则是先设定初始值<img src="http://latex.codecogs.com/gif.latex?x_0" title="x_0" />，将问题转化为求 <img src="http://latex.codecogs.com/gif.latex?f'(x_0)=0" /> 这个方程的根，然后求得的根作为下一次迭代的 _x_ 。

（2）当牛顿法用于机器学习时，目标是求解 <img src="http://latex.codecogs.com/gif.latex?L'(\theta)=0"/> 由于公式为 <img src="http://latex.codecogs.com/gif.latex?\theta^{t+1}\leftarrow \theta^t - \frac{L'(\theta)}{L''(\theta)} "/> 。牛顿法的缺点是存在求二阶偏导（海森矩阵, Hessian matrix）计算耗费高、二阶导数根本不存在则无法计算、初始值离局部极小值太远则很可能无法收敛等问题，所以在机器学习中一般不会直接使用牛顿法。

**梯度下降法（GD）**

梯度是上升最快的方向，那么如果逆着上升最快的方向就是此刻下降最快的方向，所以GD通常也称最速下降法（Steepest descent method），迭代公式为 <img src="http://latex.codecogs.com/gif.latex?\theta^{t+1}\leftarrow \theta^t - \eta  \triangledown L(\theta) "/>，其中<img src="http://latex.codecogs.com/gif.latex?\eta"/>叫学习率（Learning rate，后面会经常接触）。GD的缺点是，由于是线性收敛，所以收敛速度较慢。

**高斯-牛顿法（GN）**

GN是对牛顿法的改进，解决高维牛顿法难解决的计算问题。它用到两个矩阵，雅可比矩阵（Jacobian matrix）和海森矩阵（Hessian matrix）。其中雅可比矩阵是函数 <img src="http://latex.codecogs.com/gif.latex?f"/> 的一阶偏导矩阵，表示如下，

<img src="http://latex.codecogs.com/gif.latex?J_f=\begin{bmatrix}&space;\frac{\partial&space;f}{\partial&space;x_0}&\cdots&\frac{\partial&space;f}{\partial&space;x_n}\\&space;\vdots&\ddots&\vdots\\&space;\frac{\partial&space;f}{\partial&space;x_0}&\cdots&\frac{\partial&space;f}{\partial&space;x_n}&space;\end{bmatrix}" />

海森矩阵是函数 <img src="http://latex.codecogs.com/gif.latex?f"/> 的二阶偏导，表示如下，

<img src="http://latex.codecogs.com/gif.latex?H_f=\begin{bmatrix}\frac{\partial^2f}{\partial&space;x_0^2}&\frac{\partial^2f}{\partial&space;x_0&space;\partial&space;x_1}&...&\frac{\partial^2f}{\partial&space;x_0&space;\partial&space;x_n}\\&space;\frac{\partial^2f}{\partial&space;x_1&space;\partial&space;x_0}&\frac{\partial^2f}{\partial&space;x_1^2}&...&\frac{\partial^2f}{\partial&space;x_1&space;\partial&space;x_n}\\\vdots&\vdots&\ddots&\vdots\\&space;\frac{\partial^2f}{\partial&space;x_n&space;\partial&space;x_0}&\frac{\partial^2f}{\partial&space;x_n&space;\partial&space;x_1}&...&\frac{\partial^2f}{\partial&space;x_n^2}&space;\end{bmatrix}"/>

牛顿法公式可转换成，

<img src="http://latex.codecogs.com/gif.latex?X_{n&plus;1}=X_n-H_f(x_n)^{-1}\nabla&space;f(x_n)" />

由于 _x_ 是多维的，梯度向量在 _x_ 的第 _i_ 维上分量为

g



这样有两个好处，第一是线性、第二是只需要一阶微分。缺点是GN之反矩阵不存在时，而且在接近极值点时，Hessian矩阵变得非常小，可以类比成缩小了梯度下降法的学习率，。

**莱文贝格-马夸特方法（LM）**

其实就只是在GN基础上修改了一点，GN中，迭代公式为，而在LM方法中，。从LM的公式中可以看到，<img src="http://latex.codecogs.com/gif.latex?\lambda" /> 大的时候这种算法会接近GD，小的时候会接近GN。在LM的实际应用中，为了保证能快速稳定下降，通常会动态调整<img src="http://latex.codecogs.com/gif.latex?\lambda" />：先使用较小的<img src="http://latex.codecogs.com/gif.latex?\lambda" />，使之更接近GN，可以快速下降；

4. [wiki: Levenberg-Marquardt方法][4]
5. [wiki: 海森矩阵][5]
6. [wiki: 雅可比矩阵][6]
7. [CSDN: Levenberg-Marquardt算法浅谈][7]

[4]: (https://zh.wikipedia.org/wiki/%E8%8E%B1%E6%96%87%E8%B4%9D%E6%A0%BC%EF%BC%8D%E9%A9%AC%E5%A4%B8%E7%89%B9%E6%96%B9%E6%B3%95)
[5]: (https://zh.wikipedia.org/wiki/%E6%B5%B7%E6%A3%AE%E7%9F%A9%E9%98%B5)
[6]: (https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5)
[7]: (https://blog.csdn.net/liu14lang/article/details/53991897)
