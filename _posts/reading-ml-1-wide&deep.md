---
layout: post
title: "经典文章阅读 I · 面向推荐系统的宽深学习"
subtitle: "Wide & Deep Learning for Recommender Systems"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 经典文章阅读
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)

<center>Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, Hemal Shah</center>
<center>Google Inc.∗</center>

### 摘要

非线性特征变换的广义线性模型被广泛应用于稀疏输入的大规模回归和分类任务中。通过一个广度叉乘特征转换来表征特征间记忆功能是有效和可解释的（**特征的叉乘是逻辑回归等线性模型处理数据线性不可分问题的十分常用特征工程方法**）。深度神经网络通过从稀疏特征嵌入转换（**embedding方法，一种讲稀疏特征转换成低维稠密特征常用方法**）而成的低维稠密特征，使用更少特征工程，泛化不能直观可见的特征间联系。然而，当用户物品关系是稀疏并且高秩时，带嵌入的深度神经网络会过泛化和推荐一些不相关的物品