---
layout: post
title: "机器学习 · 监督学习篇 I"
subtitle: "监督学习是什么"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

小朋友见到了狗和猫两种动物，奶奶告诉他狗是狗，猫是猫，这是监督学习；小朋友见到了狗和猫两种动物，没人告诉他哪个是狗哪个是猫，但小朋友根据他们样子、体型等特征的不同鉴别出这是两种不同的生物，并对特征归类，这是无监督学习。

现在机器学习领域最火的就是监督学习和半监督学习，因为现在获取数据相对而言容易了一些（典型的有亚马逊 Mechanical Turk，网站上有很多分布式雇人标注数据的任务），不太可能什么标签都没有；而无监督学习一直效果都不太给力，很容易过拟合。那为何我们还需要无监督学习，因为数据获取仍需要一定的成本。监督学习、半监督学习、无监督学习关系大致可如图X表示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/learning classification.png"/>
</center>
<center>图X </center>

在传统的机器学习分类中没有提到过强化学习，而在连接主义学习中，把强化学习列为与以上三类方法并列的一类机器学习方法，个人认为可以将强化学习看成是一种通过环境内部产生样本（包括特征和标签）的监督学习。

### 频

[zhihu:什么是无监督学习？](https://www.zhihu.com/question/23194489)