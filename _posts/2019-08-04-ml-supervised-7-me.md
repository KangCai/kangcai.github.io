---
layout: post
title: "机器学习 · 监督学习篇 VII"
subtitle: "最大熵模型"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

> 最大熵模型（Maximum entropy model）

> 文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

### 一、概念

### 二、算法步骤

迭代尺度法（Generalized Iterative Scaling, GIS

改进的迭代尺度法（Improved Iterative Scaling, IIS）

### 三、表现效果

最大熵模型（Maximum Entropy Model）与逻辑回归模型（Logistic Regression Model）两者同属于广义线性模型（Generalized Linear Models），有诸多相似之处，李航老师在《统计学习方法》书中也是把两种模型放在同一章来讲的，具体来讲有以下异同：

1. 同属于广义线性模型，但逻辑回归模型假设条件概率分布是XXX，最大熵模型假设的条件概率分布是YYY。

2. 两者模型学习一般都采用极大似然估计，优化问题可以形式化为无约束的最优化问题，最优化算法有迭代尺度法、改进的迭代尺度法、梯度下降法、拟牛顿法。

**参考文献**

1. [《统计学习方法》 李航](https://book.douban.com/subject/10590856/)
2. [jianshu: 机器学习面试之最大熵模型](https://www.jianshu.com/p/e7c13002440d)