---
layout: post
title: "机器学习 · 有监督学习篇 III"
subtitle: "逻辑回归"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·有监督学习篇
---

h_\theta(x)=\frac{1}{1+e^{\theta^T x}}
<center>
<img src="https://latex.codecogs.com/gif.latex?h_\theta(x)=\frac{1}{1&plus;e^{\theta^T&space;x}}" />
</center>


\begin{aligned}
(1)\ & \left\{
\begin{matrix}
p(y=1|x;\theta)=h_\theta(x) )\\ 
\ \ \ \ \ p(y=0|x;\theta)=1-h_\theta(x)
\end{matrix}\right. \\
(2)\ & \Rightarrow  p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}\\
(3)\ & \Rightarrow L(\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta) \\
(4)\ & \Rightarrow L(\theta)=\prod_{i=1}^{m}(h_\theta(x^{(i)}))^y^{(i)}(1-h_\theta(x^{(i)}))^{1-y^{(i)}} \\
\end{aligned}

<center>
<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;(1)\&space;&&space;\left\{&space;\begin{matrix}&space;p(y=1|x;\theta)=h_\theta(x)&space;)\\&space;\&space;\&space;\&space;\&space;\&space;p(y=0|x;\theta)=1-h_\theta(x)&space;\end{matrix}\right.&space;\\&space;(2)\&space;&&space;\Rightarrow&space;p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}\\&space;(3)\&space;&&space;\Rightarrow&space;L(\theta)=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta)&space;\\&space;(4)\&space;&&space;\Rightarrow&space;L(\theta)=\prod_{i=1}^{m}(h_\theta(x^{(i)}))^y^{(i)}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}&space;\\&space;\end{aligned}"/>
</center>