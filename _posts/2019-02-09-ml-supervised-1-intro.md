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

> 总览篇介绍完后，接下来就是在监督学习、深度学习、强化学习、无监督学习、半监督学习中选一个主题作为第二个大篇章，首先深度学习跟其它四类不在一个位面上：深度学习是根据模型的结构上的差异化形成的机器学习的一个分支，而其它四类是连接主义学习的四大类别，深度学习跟四类都是有都分重叠关系。所以对于以上五个主题，无论是从知识广度还是应用范围来排优先级都不好排，思来想去还是结合 “领域热度”、“历史发展顺序”、“知识延展顺序” 这三个方面来排篇章顺序：监督学习（发展最早、热度较高）、深度学习（热度最高）、强化学习（热度较高）、无监督学习（基础知识）、半监督学习（延展知识），第一个介绍的篇章是监督学习篇，本文主要对监督学习进行概括性介绍。
> 算法。文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/learning classification2.png"/>
</center>
<center>图1 机器学习分类</center>

传统的机器学习分类中通常是：监督学习（Supervised learning）和无监督学习（Unsupervised learning），还有一种结合监督学习和无监督学习的中间类别，称之为半监督学习（Semi-supervised Learning）。传统的机器学习分类没有提到过强化学习，而在连接主义学习中，把强化学习（Reinforcement learning）作为与以上三类方法并列的一类机器学习方法，个人认为可以把强化学习看成是一种通过环境内部产生样本（包括特征和标签）的监督学习。

**监督学习和无监督学习很好区分：是否有监督（supervised），就看输入数据是否有标签（label），输入数据有标签，则为有监督学习，没标签则为无监督学习。** 一个话糙理不糙的例子是：你小时候见到了狗和猫两种动物，奶奶告诉你狗是狗、猫是猫，你学会了辨别，这是监督学习；你小时候见到了狗和猫两种动物，没人告诉你哪个是狗、哪个是猫，但你根据他们样子、体型等特征的不同鉴别出这是两种不同的生物，并对特征归类，这是无监督学习。监督学习、半监督学习、无监督学习关于数据标注完整度的比较大致可如图2表示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/learning classification.png"/>
</center>
<center>图2 数据标注完整度从高到低</center>

图2中监督学习的数据标注完整度是100%；无监督学习使用没有标签的数据，标注完整度是0%；**半监督学习使用 “有标签数据+无标签数据” 混合成的数据；半监督聚类里数据的标签不是确定性的**，举个例子，标签可能是 “样本 A 的分类不是 C 类”，或者是 “A、B 两类中的一类” 这些形式。

### 一、监督学习模型

如《总览篇 V》中所述，模型通常有两种分类方式：第一种是**按模型形式分类：概率模型（Probabilistic Model）和 非概率模型（Non-probabilistic Model）**；第二种是**按是否对观测变量的分布建模分类：判别模型（Discriminative Model）和 生成模型（Generative Model）**。这两种分类方法事实上把所有模型划分成了三类，可用图1清楚地表示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/Model classification.png"/>
</center>
<center>图1 模型分类示意图</center>

其中，每种类别的代表性模型如下：

I. 非概率模型（Non-probabilistic Model），顾名思义，不是根据概率进行分类的，而是直接通过计算判别超平面的方式进行分类。实例：感知机（单层神经网络，Perceptron）、多层感知机（MLP）、支持向量机（SVM）、K近邻（KNN）

II. 概率判别模型（Probalilistic Discriminative Model），与非概率模型相同的是，都是判别模型，必须通过多类数据之间对比才能建模；不同的是，它是计算后验概率来间接获得判别超平面的。实例：逻辑回归（LR）、最大熵模型（ME）、条件随机场（CRF）

III. 生成模型（Generative Model），对某一类数据自己就可以单独建模。首先是对某类数据的联合概率P(x,y)进行建模，如果是在聚类任务中，到这一步就结束了；但如果在分类任务中，还需要根据贝叶斯公式算出后验概率实现分类。实例：高斯判别分析（GDA）、朴素贝叶斯（NB）、受限玻尔兹曼机（RBM）、隐马尔科夫模型（HMM）

### 二、如何选择合适的监督学习算法

这个问题的答案取决于许多的因素，其中包括：

1. 数据的维度（大还是小），数据值的形式（连续还是离散）；
2. 你可以利用的计算资源（CPU或者GPU的计算能力）；
3. 要求对模型可解释性的强弱。虽然学习如何权衡输入变量的复杂组合能够带来更准确的预测，但它使解释机器学习模型变得真的很困难。可被解释的预测模型生成的决策，是由原始输入变量带来的，而不是输入变量的任意高阶组合、缩放、加权组合带来的，所以对特征进行操作越直观的模型可解释性越强。

根据以上因素，可以选择一个直观上比较合适的模型进行建模。那么具体如何利用这一先验知识，指导我们去选择算法模型，帮助我们少走一点弯路呢？下面图3是一个比较经典的模型选择流程图，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/supervised learning model cheeting sheet.png"/>
</center>
<center>图3 监督学习模型选择表</center>

首先毫无疑问，我们可以根据数据值的因变量，或者称之为标签，是连续值还是离散值，将监督学习问题分为分类问题和回归问题，其中分类问题的标签是离散值，回归问题的标签是连续值，

##### 2.1 分类模型

|  | 训练复杂度 | 预测复杂度 | 可解释性 | 
| :-----------:| :----------: | :----------: | :----------: |:----------: |
| 朴素贝叶斯 |   |  | |
| 逻辑回归 | <img src="https://latex.codecogs.com/gif.latex?O(M)" /> | <img src="https://latex.codecogs.com/gif.latex?O(M)" />  |   |   |
| 支持向量机 | <img src="https://latex.codecogs.com/gif.latex?O(N^2M^2)" /> | <img src="https://latex.codecogs.com/gif.latex?O(NM^2)" /> | 弱 |
| 神经网络 |  |   |   |   |
| 高斯判别模型 |  |  |    |
| 决策树 |   |   |   |   |
| 随机森林 |   |   |   |   |
| GBDT |   |   |   |   |
| XGBoost |   |   |   |   |

##### 2.2 回归模型

注重效率的分类模型

（表格）朴素贝叶斯、决策树、逻辑回归、线性SVM，

注重准确率的分类模型

（表格）核函数SVM、随机森林、神经网络、GBDT、XGBoost

注重效率的回归模型

（表格）

注重准确率的回归模型

（表格）


machine learning algorithm cheat sheet
https://blog.csdn.net/gitchat/article/details/78913235
https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/

微软 Azure 算法流程图
来源： https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-algorithm-cheat-sheet

https://blog.csdn.net/tkkzc3E6s4Ou4/article/details/80000439


**参考文献**

[zhihu:什么是无监督学习？](https://www.zhihu.com/question/23194489)