---
layout: post
title: "机器学习 · 监督学习篇 IV"
subtitle: "K近邻"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

K最近邻(k-Nearest Neighbor，KNN)分类算法，一种很 “直白” 的算法，它的算法思路是，如果一个样本在特征空间中 K 个最相似（度量距离最近）的样本中大多数属于某一类别，那么这个样本也属于这个类别，以下图的情况为例，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/knn_1.jpg"/>
</center>

，如上图的情况，如果K=3，那么离绿色点最近的有2个红色三角形和1个蓝色的正方形，这3个点投票，于是绿色的这个待分类点属于红色的三角形；
如果K=5，那么离绿色点最近的有2个红色三角形和3个蓝色的正方形，这5个点投票，于是绿色的这个待分类点属于蓝色的正方形。我们可以看到，
 KNN 本质是基于一种数据统计的方法。

KNN 是一种 memory-based learning，也叫 instance-based learning，没有显式的前期训练过程，而是在预测时把数据集加载到内存后，根据
数据样本来进行分类。个人认为朴素贝叶斯模型与 KNN 在这一点上很相似，也是没有显式的训练过程。

KNN 与 K-Means 的区别

| | KNN | K-Means |
| :-----------:| :----------: |:----------: | 
| 类型 | 监督学习、分类 | 无监督学习、聚类 |
| 训练过程 | 无显式 | 显式 |
| k的含义 | 预测过程中参考的样本数目 | 训练过程中聚类的类别数 |
| | 11 |

[Kmeans算法与KNN算法的区别](https://www.cnblogs.com/peizhe123/p/4619066.html)