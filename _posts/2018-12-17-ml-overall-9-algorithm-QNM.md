---
layout: post
title: "机器学习 · 总览篇 IX"
subtitle: "三要素之算法 - 牛顿法与拟牛顿法"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

> 在机器学习中，牛顿法与拟牛顿法在解决目标函数的最优化问题方面也起着重要的作用。本篇是机器学习三要素之算法的第二篇，也是三要素介绍的最后一篇。最近的5篇文章完整地介绍了机器学习的三要素，对三要素的掌握对于机器学习的学习至关重要，所有机器学习方法的想法和实现都离不开这三个要素。

> 算法。文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

与梯度下降法一样，在解决机器学习目标函数的最优化问题时，牛顿法与拟牛顿法也起着重要的作用。本文将分三节分别介绍 牛顿法一族、拟牛顿法一族 以及 两者与梯度下降法的比较。

### 一、牛顿法

##### 1.1 原始的牛顿法（Newton's method）

**用于求方程解**

牛顿法用于求方程解，即<img src="http://latex.codecogs.com/gif.latex?f(x)=0" />问题时，首先是设定初始值<img src="http://latex.codecogs.com/gif.latex?x_0" title="x_0" />，获得此处的导数 <img src="http://latex.codecogs.com/gif.latex?f'(x_0)" />，计算下一个 x 值，<img src="https://latex.codecogs.com/gif.latex?x_1=x_0-\frac{f(x_0)}{f'(x_0)}"/>，重复以下迭代过程，

<center>
<img src="https://latex.codecogs.com/gif.latex?x_{n&plus;1}=x_n-\frac{f(x_n)}{f'(x_n)}"/>
</center>

上述公式就是牛顿法的迭代公式，wiki 动图可以形象表示出牛顿法的迭代过程，如图1所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/NewtonIteration_Ani.gif"/>
</center>
<center>图1 牛顿迭代法求方程 f(x)=0 的解</center>

简单地介绍一下牛顿法迭代公式的推导过程，首先将方程泰勒展开，

<center>
<img src="https://latex.codecogs.com/gif.latex?f(x)=f(x_0)&plus;\frac{f'(x_0)}{1!}(x-x_0)&plus;\frac{f''(x_0)}{2!}(x-x_0)^2&plus;\frac{f'''(x_0)}{3!}(x-x_0)^3&plus;..." />
</center>

，忽略二次以上的项，

<center>
<img src="https://latex.codecogs.com/gif.latex?f(x)=f(x_0)&plus;\frac{f'(x_0)}{1!}(x-x_0)&plus;\frac{f''(x_0)}{2!}(x-x_0)^2" />
</center>

两边对 x 求导，注意 x0 相关的值当作常量处理，

<center>
<img src="https://latex.codecogs.com/gif.latex?f'(x)=f'(x_0)&plus;f''(x_0)(x-x_0)" />
</center>

根据微积分的性质，f(x) 取最小值时，有 f′(x)=0 ，代入上面的式子有：

<center>
<img src="https://latex.codecogs.com/gif.latex?x=x_0-\frac{f'(x_0)}{f''(x_0)}" />
</center>

**用于最优化问题**

当牛顿法用于机器学习的最优化问题时，目标是求解 <img src="http://latex.codecogs.com/gif.latex?L'(\theta)=0"/>。可以类比于求方程解，迭代公式为 

<center>
<img src="http://latex.codecogs.com/gif.latex?\theta^{t+1}\leftarrow \theta^t - \frac{L'(\theta)}{L''(\theta)} "/> 。
</center>

当函数输出为1维时，L'(θ)为 L 关于 θ 的梯度向量，而当函数输出为 m 维时，即<img src="https://latex.codecogs.com/gif.latex?L(\theta)=(L_1(\theta),L_2(\theta),\&space;...\&space;,L_m(\theta))"/>，则 L'(θ) 是 L 关于 θ 的一阶偏导构成的矩阵，该矩阵在19世纪初由德国数学家雅可比提出且被命名为雅可比矩阵，θ 表示成向量形式 <img src="https://latex.codecogs.com/gif.latex?(x_0,x_1,\&space;...\&space;,x_n)"/>，则雅可比矩阵表示为

<center>
<img src="http://latex.codecogs.com/gif.latex?J_L=\begin{bmatrix}&space;\frac{\partial&space;L_1}{\partial&space;x_0}&\cdots&\frac{\partial&space;L_1}{\partial&space;x_n}\\&space;\vdots&\ddots&\vdots\\&space;\frac{\partial&space;L_m}{\partial&space;x_0}&\cdots&\frac{\partial&space;L_m}{\partial&space;x_n}&space;\end{bmatrix}" />
</center>

<img src="http://latex.codecogs.com/gif.latex?L''(\theta)" /> 是函数 L 关于 θ 的二阶偏导构成的矩阵，类似于雅可比矩阵，该矩阵在19世纪由德国数学家海森提出且被命名为海森矩阵，海森矩阵表示为

<center>
<img src="http://latex.codecogs.com/gif.latex?H_L=\begin{bmatrix}\frac{\partial^2L}{\partial&space;x_0^2}&\frac{\partial^2L}{\partial&space;x_0&space;\partial&space;x_1}&...&\frac{\partial^2L}{\partial&space;x_0&space;\partial&space;x_n}\\&space;\frac{\partial^2L}{\partial&space;x_1&space;\partial&space;x_0}&\frac{\partial^2L}{\partial&space;x_1^2}&...&\frac{\partial^2L}{\partial&space;x_1&space;\partial&space;x_n}\\\vdots&\vdots&\ddots&\vdots\\&space;\frac{\partial^2L}{\partial&space;x_n&space;\partial&space;x_0}&\frac{\partial^2L}{\partial&space;x_n&space;\partial&space;x_1}&...&\frac{\partial^2Lf}{\partial&space;x_n^2}&space;\end{bmatrix}"/>
</center>

，将雅可比矩阵 J 和海森矩阵 H 带入原迭代公式，由于在迭代公式的迭代量中，海森矩阵是除数，所以实际上是乘以海森矩阵的逆，迭代公式可表示为，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta^{t&plus;1}\leftarrow\theta^t-H_L^{-1}(\theta^t)J(\theta^t)"/>
</center>

**优点：二阶收敛，收敛速度快**

二阶收敛的牛顿法相比于一阶收敛的梯度下降法，收敛速度更快，因为二阶收敛还额外考虑了下降速度变化的趋势。从几何上解释，牛顿法就是用一个二次曲面去拟合你当前所处位置的局部曲面，而梯度下降法是用一个平面去拟合当前的局部曲面，通常情况下，二次曲面的拟合会比平面更好，所以牛顿法选择的下降路径会更符合真实的最优下降路径，路径找对了，下降速度就更快。

**缺点：难以计算**

牛顿法的迭代过程中，每一步都需要求解目标函数的Hessian矩阵的逆矩阵，计算比较复杂甚至是无法计算，这个问题很严重，所以在机器学习中甚至都不会直接使用牛顿法。

##### 1.2 高斯-牛顿法（Gauss-Newton algorithm，GN）

**GN 是牛顿法求解最小二乘问题时的派生出的特殊求解方法**。回顾一下牛顿法的迭代公式，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta^{t&plus;1}\leftarrow\theta^t-H_L^{-1}(\theta^t)J(\theta^t)"/>
</center>

其中，对于最小二乘问题，L 是如下形式，

<center>
<img src="https://latex.codecogs.com/gif.latex?L(\theta)=\frac{1}{2}r(\theta)^Tr(\theta)=\frac{1}{2}\sum_{i=1}^{m}[r_i(\theta)]^2,&space;\quad&space;m\geq&space;n"/>
</center>

其中 r 为最小二乘问题的残差，即 L 函数值与 groudtruth 的插值。L 关于 θ 的雅可比矩阵 J 和海森矩阵 H 可用 r 表示为，

<center>
<img src="https://latex.codecogs.com/gif.latex?J_j=2\sum_{i=1}^{m}r_i&space;\frac{\partial&space;r_i}{\partial&space;x_j}=2J_r^Tr,&space;\quad&space;H_{jk}=2\sum_{i=1}^{m}(\frac{\partial&space;r_i}{\partial&space;x_j}\frac{\partial&space;r_i}{\partial&space;x_k}&plus;r_i\frac{\partial&space;^2r_i}{\partial&space;x_j\partial&space;x_k})" />
</center>

高斯牛顿法（GN）通过舍弃用海森矩阵的二阶偏导数实现，也就有对海森矩阵的近似计算，

<center>
<img src="https://latex.codecogs.com/gif.latex?H_{jk}\approx&space;2\sum_{i=1}^m&space;J_{ij}J_{ik}=2J_r^TJ_r"/>
</center>

带入到牛顿法的迭代公式，则 GN 的迭代公式表示为，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta^{t&plus;1}\leftarrow\theta^t-(J_r^TJ_r)^{-1}J_r^Tr"/>
</center>

**优点：计算量小**

主要问题是因为牛顿法中Hessian矩阵 H 中的二阶信息项通常难以计算或者花费的工作量很大，又因为在计算梯度时已经得到一阶偏导 J，这样 H 中的一阶信息项几乎是现成的。鉴于此，为了简化计算，获得有效算法，我们可用一阶导数信息逼近二阶信息项。注意这么干的前提是，残差 r 接近于零或者接近线性函数从而接近与零时，二阶信息项才可以忽略，通常称为“小残量问题”，最典型的就是最小二乘问题，否则高斯牛顿法不收敛。

**缺点：收敛要求较为严格**

对于残量 r 值较大的问题，收敛速度较慢；对于残量很大的问题，不收敛；不能保证全局收敛（收敛性与初始点无关，则是全局收敛；当初值靠近最优解时才收敛，则是局部收敛；牛顿法和 GN 都是局部收敛）。

##### 1.3 莱文贝格-马夸特方法（Levenberg–Marquardt algorithm，LM）

**LM 同时结合了梯度下降法（GD）稳定下降的优点和高斯牛顿法（GN）在极值点附近快速收敛的优点，同时避免了 GD 和 GN 相应的缺点**。GN 的迭代公式为，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta^{t&plus;1}\leftarrow\theta^t-(J_r^TJ_r)^{-1}J_r^Tr"/>
</center>

而在 LM 方法的迭代公式为，

<center>
<img src="https://latex.codecogs.com/gif.latex?\theta^{t&plus;1}\leftarrow\theta^t-(J_r^TJ_r+\lambda I)^{-1}J_r^Tr"/>
</center>

，从 LM 的公式中可以看到，λ 大的时候这种算法会接近 GD（梯度下降法），小的时候会接近 GN（高斯牛顿法） 。在 LM 的实际应用中，为了保证能快速稳定下降，通常会根据 L 函数值真实减少量与预测减少量 ρ 来动态调整 λ ：

1. 当 0 < ρ < 阈值，则不改变 λ ；
2. 当 ρ > 阈值，说明近似效果很好，则增大 λ；
3. 当 L 函数值是增加，即 ρ < 0，说明近似效果很差，则减小 λ。

### 二、拟牛顿法（Quasi-Newton Methods）

**拟牛顿法的本质思想是在牛顿法的基础上，使用迭代法来获得 Hessian 矩阵的逆矩阵** ：这样一来，只需要最开始求一次逆，后面就可以通过迭代的方法来获取每一次需要使用的 Hessian 矩阵的逆矩阵，从而简化了迭代运算的复杂度。

拟牛顿法于20世纪50年代由美国 Argonne 国家实验室的物理学家 W.C.Davidon 所提出来，这种算法在当时看来是非线性优化领域最具创造性的发明之，而且随后该算法被证明远比其他方法快速和可靠，使得非线性优化这门学科在一夜之间突飞猛进。机器学习中常见的拟牛顿法有 DFP 法、BFGS 法、 L-BFGS法。

那么拟牛顿法一族，包括DFP 法、BFGS 法、 L-BFGS法，是如何构造 Hessian 矩阵的逆矩阵的迭代公式呢？回到1.1节中牛顿法迭代公式的推导过程，二阶泰勒展开式对 x 进行求导得，

<center>
<img src="https://latex.codecogs.com/gif.latex?f'(x)=f'(x_{k+1})&plus;f''(x_{k+1})(x-x_{k+1})" />
</center>

将 <img src="https://latex.codecogs.com/gif.latex?x=x_k"/> 带入公式有，

<center>
<img src="https://latex.codecogs.com/gif.latex?f''(x_{k&plus;1})^{-1}(f'(x_{k&plus;1})-f'(x_k))=x_{k&plus;1}-x_k" />
</center>

令 <img src="https://latex.codecogs.com/gif.latex?H_{k&plus;1}^{-1}=f''(x_{k&plus;1})^{-1},&space;\&space;y_k=f'(x_{k&plus;1})-f'(x_k),\&space;s_k=x_{k&plus;1}-x_k"/>，则公式可以简单表示为，

<center>
<img src="https://latex.codecogs.com/gif.latex?H_{k&plus;1}^{-1}y_k=s_k" />
</center>

##### 1.1 DFP法（Davidon-Fletcher-Powell algorithm）

**DFP法的核心思想是直接构造 Hessian 矩阵的逆矩阵的迭代公式**。继续上文公式的推导，先假海森矩阵的逆矩阵的迭代公式为 <img src="https://latex.codecogs.com/gif.latex?H_{k&plus;1}^{-1}=H_k^{-1}&plus;E_k" />，DFP 法的目标就是求这个 <img src="https://latex.codecogs.com/gif.latex?E_k" />，将 <img src="https://latex.codecogs.com/gif.latex?E_k=\alpha&space;u_ku_k^T&plus;\beta&space;v_kv_k^T" /> 代入上式有，

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&\(H_k^{-1}&plus;\alpha&space;u_ku_k^T&plus;\beta&space;v_kv_k^T)y_k=s_k&space;\\&space;\Rightarrow&space;\&space;&\alpha(u_k^Ty_k)u_k&plus;\beta(v_k^Ty_k)v_k=s_k-H_k^{-1}y_k&space;\end{aligned}"  />
</center>

然后假设<img src="https://latex.codecogs.com/gif.latex?u_k=rH_k^{-1}y_k,v_k=\theta&space;s_k"/>，代入上式，

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&&space;\Rightarrow&space;\&space;\alpha[((rH_k^{-1}y_k)^Ty_k)(rH_k^{-1}y_k)&plus;\beta((\theta&space;s_k)^Ty_k)(\theta&space;s_k)=s_k-H_k^{-1}y_k&space;\\&space;&&space;\Rightarrow&space;\&space;[\alpha&space;r^2(y_k^TH_k^{-1}y_k)&plus;1](H_k^{-1}y_k)&plus;[\beta\theta^2(s_k^Ty_k)-1]s_k=0&space;\end{aligned}"/>
</center>

令 <img src="https://latex.codecogs.com/gif.latex?\alpha&space;r^2(y_k^TH_k^{-1}y_k)&plus;1=0,\&space;\beta\theta^2(s_k^Ty_k)-1=0"/> 使上式满足，则有

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\alpha&space;r^2&=-\frac{1}{y_k^T&space;H_k^{-1}&space;y_k}\\&space;\beta&space;\theta^2&=&space;\frac{1}{s_k^T&space;y_k}&space;\end{aligned}"/>
</center>

整合上述 u, v, α, β 相关的表达式，带入最终 DFP 法得到的 Hessian 矩阵的逆矩阵的迭代公式，

<center>
<img src="https://latex.codecogs.com/gif.latex?H_{k&plus;1}^{-1}=H_k^{-1}-\frac{H_k^{-1}&space;y_k&space;y_k^T&space;H_k^{-1}}{y_k^T&space;H_k^{-1}&space;y_k}&plus;\frac{s_k&space;s_k^T}{s_k^Ty_k}"/>
</center>

##### 1.2 BFGS法（Broyden–Fletcher–Goldfarb–Shanno algorithm）

**BFGS 算法与 DFP 算法的区别在于，迭代公式结果不一样，对 Hessian 矩阵的逆矩阵的构造方法也不一样：其中，DFP算法是直接构造，而BFGS是分两步走：先求 Hessian 矩阵的迭代公式，然后根据Sherman-Morrison公式转换成Hessian 矩阵的逆矩阵的迭代公式**。

首先求 Hessian 矩阵的迭代公式，虽然 DFP 法求得是 Hessian 逆矩阵的迭代公式，但其实可以用同样的推导过程求 Hessian 矩阵的迭代公式，如下表示，

<center>
<img src="https://latex.codecogs.com/gif.latex?H_{k&plus;1}=H_k-\frac{H_k&space;y_k&space;y_k^T&space;H_k}{y_k^T&space;H_k&space;y_k}&plus;\frac{s_k&space;s_k^T}{s_k^Ty_k}"/>
</center>

然后根据 Sherman-Morrison 公式，即对于任意 n x n 的 [非奇异矩阵](https://baike.baidu.com/item/%E9%9D%9E%E5%A5%87%E5%BC%82%E7%9F%A9%E9%98%B5/4114613) ，u、v 都是 n 维向量，若 <img src="https://latex.codecogs.com/gif.latex?1&plus;v^TA^{-1}u\neq&space;0" />，则

<center>
<img src="https://latex.codecogs.com/gif.latex?(A&plus;uv^T)^{-1}&space;=&space;A^{-1}-\frac{(A^{-1}u)(v^TA^{-1})}{1&plus;v^TA^{-1}u}" />
</center>

该公式描述了在矩阵 A 发生某种变化时，如何利用之前求好的逆，求新的逆，而这个性质正是我们需要的，接下来通过该式子将 Hessian 矩阵的迭代公式改造一下，得到 Hessian 矩阵的逆矩阵的迭代公式可表示为，

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&H^{-1}_{n&plus;1}=(I&space;-&space;\rho_n&space;y_n&space;s_n^T)&space;H^{-1}_n&space;(I&space;-&space;\rho_n&space;s_n&space;y_n^T)&space;&plus;&space;\rho_n&space;s_n&space;s_n^T&space;\\&space;&&space;where&space;\&space;\rho_n&space;=&space;(y_n^T&space;s_n)^{-1}&space;\end{aligned}" />
</center>

**优点**

BFGS 法相比于 DFP 法，对Hessian 矩阵的逆矩阵近似误差更小，原因是因为 BFGS 具有自校正的性质(self-correcting property)。通俗来说，如果某一步 BFGS 对 Hessian矩阵的逆矩阵估计偏了，导致优化变慢，那么BFGS会在较少的数轮迭代内校正。对证明感兴趣可以参考[《A Tool for the Analysis of Quasi-Newton Methods with Application to Unconstrained Minimization》](https://epubs.siam.org/doi/10.1137/0726042)。

**缺点**

拟牛顿法都共同具有的缺点，即对 Hessian 矩阵的逆矩阵的近似存在误差。

##### 1.3 L-BFGS法（Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm）

**L-BFGS 法是 BFGS 算法的一个小改进，本质是用准确度下降的少量代价来换取大量空间的节省，它对 BFGS 算法进行了近似，不存储完整的逆矩阵**。回顾一下 BFGS 的迭代公式，

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&H^{-1}_{n&plus;1}=(I&space;-&space;\rho_n&space;y_n&space;s_n^T)&space;H^{-1}_n&space;(I&space;-&space;\rho_n&space;s_n&space;y_n^T)&space;&plus;&space;\rho_n&space;s_n&space;s_n^T&space;\\&space;&&space;where&space;\&space;\rho_n&space;=&space;(y_n^T&space;s_n)^{-1}&space;\end{aligned}" />
</center>

L-BFGS 不存储逆矩阵，而是存储最新 m 个上面公式中的 s 和 y 向量，即 <img src="https://latex.codecogs.com/gif.latex?\{s_i\}\{y_i\},\&space;i=n-m&plus;1,\&space;...\&space;,n"/>，从而使存储空间复杂度开销从 <img src="https://latex.codecogs.com/gif.latex?O(N^2)"/> 降低到 <img src="https://latex.codecogs.com/gif.latex?O(mN)"/>。

### 三、梯度下降法、牛顿法、拟牛顿法的比较

梯度下降法一族（如 SGD、各种 Adaptive SGD）、牛顿法一族（如 Gauss-Newton Method，LM 法）、拟牛顿法一族（如 DFP 法、L-BFGS 法）是机器学习中最常见的三大类迭代法。用表格可以直观地对比较三种迭代大类的差别，分别从 迭代公式 、收敛性 、在机器学习中擅长的应用场景 以及 在实际应用中最具代表性的实现，这4个角度进行比较，如下所示，

|  | 迭代公式 | 收敛性 | 擅长场景 | 应用代表 | 
| :-----------:| :----------: | :----------: | :----------: | :----------: |
| 梯度下降法 | <img src="https://latex.codecogs.com/gif.latex?\theta_{k+1}\leftarrow\theta_k-\alpha \bigtriangledown L_k"/> | 全局收敛、一阶收敛| 神经网络 | SGD、RMSprop、Adam |
| 牛顿法 | <img src="https://latex.codecogs.com/gif.latex?\theta_{k+1}\leftarrow\theta_k-H_k^{-1} \bigtriangledown L_k"/> | 局部收敛、二阶收敛 | 非线性最小二乘问题 | LM 法 | 
| 拟牛顿法 | <img src="https://latex.codecogs.com/gif.latex?\theta_{k&plus;1}\leftarrow\theta_k-H_k^{-1}&space;\bigtriangledown&space;L_k&space;\\&space;where,\&space;H_k^{-1}=H_{k-1}^{-1}&plus;E_{k-1}"/>| 局部收敛、二阶收敛 | 逻辑回归 | L-BFGS |

事实上，在机器学习实际应用中，还是梯度下降法（比如SGD），特别是自适应学习率的梯度下降法（比如RMSprop、Adam）更实用也更常用，应用局限性也最低，在更多的情况下能稳定收敛。

至此，最近的5篇文章完整地介绍了机器学习的三要素，对三要素的掌握对于机器学习的学习至关重要，所有机器学习方法的想法和实现都离不开这三个要素。

1. [wiki: 牛顿法](https://zh.wikipedia.org/wiki/%E7%89%9B%E9%A1%BF%E6%B3%95)
2. [wiki: Levenberg-Marquardt方法](https://zh.wikipedia.org/wiki/%E8%8E%B1%E6%96%87%E8%B4%9D%E6%A0%BC%EF%BC%8D%E9%A9%AC%E5%A4%B8%E7%89%B9%E6%96%B9%E6%B3%95)
3. [wiki: 海森矩阵](https://zh.wikipedia.org/wiki/%E6%B5%B7%E6%A3%AE%E7%9F%A9%E9%98%B5)
4. [wiki: 雅可比矩阵](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5)
5. [csdn: Levenberg-Marquardt算法浅谈](https://blog.csdn.net/liu14lang/article/details/53991897)
6. [zhihu: DFP与BFGS算法的比较？](https://www.zhihu.com/question/34873977/answer/242695668)
7. [zhihu: 梯度下降or拟牛顿法？](https://www.zhihu.com/question/46441403)
8. [csdn: 最全的机器学习中的优化算法介绍](https://blog.csdn.net/qsczse943062710/article/details/76763739)
9. [csdn：Gauss-Newton算法学习](https://blog.csdn.net/jinshengtao/article/details/51615162 
)
10. [csdn: 优化算法——拟牛顿法之DFP算法](https://blog.csdn.net/google19890102/article/details/45848439)
11. [csdn: 拟牛顿法（DFP、BFGS、L-BFGS）](https://blog.csdn.net/songbinxu/article/details/79677948
)
12. [cnblog: 常见的几种最优化方法](http://www.cnblogs.com/maybe2030/p/4751804.html)


