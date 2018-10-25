---
layout: post
title: "机器学习-总览篇(2) 统计推断"
subtitle: "统计推断中的频率学派和贝叶斯学派"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
---

> 天下难事，必作于易

统计推断作为重要的机器学习基础，对它的了解十分必要，否则我们做机器学习只是在黑盒操作，对其原理和结果难以解释。而统计推断又是统计学的一个庞大的分支，
我们可以从频率学派和贝叶斯学派的争论去了解统计推断，

* 频率学派（Frequentist）- 最大似然估计（MLE, Maximum Likelihood Estimation）
* 贝叶斯学派（Bayesians）- 最大后验估计（MAP, Maximum A Posteriori）
* 频率学派，20世纪初期建立，在之后的整个20世纪基本主宰了统计学，代表人费舍尔（Fisher）、K皮尔逊（Karl Pearson）、内曼（Neyman），
费舍尔提出极大似然估计方法（Maximum Likelihood Estimation，MLE）和多种抽样分布，K皮尔逊提出Pearson卡方检验、Pearson相关系数，
内曼提出了置信区间的概念，和K卡尔逊的儿子ES皮尔逊提出了假设检验的内曼-皮尔森引理；
* 贝叶斯学派（Bayesians），20世纪30年代建立，快速发展于20世纪50年代（计算机诞生后），它的理论基础由17世纪的贝叶斯（Bayes）提出了，
他提出了贝叶斯公式，也称贝叶斯定理，贝叶斯法则。贝叶斯方法经过高斯（Gauss）和LapLace（拉普拉斯）的发展，在19世纪主宰了统计学。
* 抽象地说，两种学派的主要差别在于探讨「不确定性」这件事的立足点不一样，频率学派试图对「事件」本身建模，认为「事件本身就具有客观的不确定性」；
贝叶斯学派不去试图解释「事件本身的随机性」，而是从观察事件的「观察者」角度出发，认为不确定性来源于「观察者」的「知识不完备」，在这种情况下，
通过已经观察到的「证据」来描述最有可能的「猜的过程」，因此，在贝叶斯框架下，同一件事情对于知情者而言就是「确定事件」，对于不知情者而言就是「随机事件」，
随机性并不源于事件本身是否发生，而只是描述观察者对该事件的知识状态。
* 具体来说，两种学派的主要差别是在对参数空间的认知上，即参数的可能取值范围。频率学派认为存在唯一的真实常数参数，观察数据都是在这个参数下产生的，由于不知道
参数到底是哪个值，所以就引入了最大似然（Maximum Likelihood）和置信区间（confidence interval）来找出参数空间中最可能的参数值；
贝叶斯学派认为参数本身存在一个概率分布，并没有唯一真实参数，参数空间里的每个值都可能是真实模型使用的参数，区别只是概率不同，
所以就引入了先验分布（prior distribution）和后验分布（posterior distribution）来找出参数空间每个参数值的概率。
* 现状是仍在互相争论，也在发展中互相借鉴，经过。

用一个例子去更好地理解两种学派的对应两种方法，首先是一些名词的解释

* θ: 模型参数
* x: 观察样本点
* 似然函数: 在字典中，似然（likelihood）和概率（probability）是差不多的意思，但在统计学里，似然函数和概率函数的含义却不同，对于形式为<a href="http://www.codecogs.com/eqnedit.php?latex=P(x|\theta)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P(x|\theta)" title="P(x|\theta)" /></a>
的函数，如果θ是已知的，x是变量，这个函数叫做概率函数（probability function），描述的是不同的样本x出现的概率是多少; 如果x是已知的，θ是变量，
这个函数叫做似然函数（likelihood function），描述的是不同的模型参数下，出现x这个样本的概率是多少。
举个抛硬币的简单例子，我们先假设硬币均匀，硬币出现朝上概率p=0.5，那么出现两次朝上的概率是0.25，这个0.25是似然函数值，
如果之前我们假设的是硬币不均匀，朝上概率p=0.6，那么出现两次朝上概率就成了0.36，似然函数值越大能**表征**θ成立的概率越大，**但一定要注意似然函数值并不等于出现样本x时θ成立的概率**，即更关注值的大小关系，而非值本身。
* 最大似然估计：其实上面抛两次硬币的例子就进行了似然估计，其曲线可表示为

而且当p=0.5时，似然估计值为0.25；p=0.6时，似然估计值为0.36；在p=1.0时能取到最大似然估计值1.0。

参考文献

1. [WIKI: 贝叶斯][1]
2. [WIKI: 费舍尔][2]
3. [WIKI: 皮尔逊][3]
4. [WIKI: 内曼][4]
5. [丁以华.贝叶斯方法的发展及其存在问题[J].质量与可靠性,1986(01):29-31.][5]
7. [知乎: 如何理解 95% 置信区间？][7]
8. [知乎: 贝叶斯学派与频率学派有何不同？][8]
9. [CSDN博客: 详解最大似然估计（MLE）、最大后验概率估计（MAP），以及贝叶斯公式的理解][9]

[1]: (https://zh.wikipedia.org/wiki/%E6%89%98%E9%A9%AC%E6%96%AF%C2%B7%E8%B4%9D%E5%8F%B6%E6%96%AF)
[2]: (https://zh.wikipedia.org/wiki/%E7%BE%85%E7%B4%8D%E5%BE%B7%C2%B7%E6%84%9B%E7%88%BE%E9%BB%98%C2%B7%E8%B2%BB%E9%9B%AA)
[3]: (https://zh.wikipedia.org/wiki/%E5%8D%A1%E5%B0%94%C2%B7%E7%9A%AE%E5%B0%94%E9%80%8A)
[4]: (https://zh.wikipedia.org/wiki/%E8%80%B6%E6%97%A5%C2%B7%E5%86%85%E6%9B%BC)
[5]: (http://xueshu.baidu.com/s?wd=paperuri%3A%287eefad3052335afda45d48e995abcd8c%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fkns.cnki.net%2FKCMS%2Fdetail%2Fdetail.aspx%3Ffilename%3DZNYZ198601014%26dbname%3DCJFD%26dbcode%3DCJFQ&ie=utf-8&sc_us=1462943048446069877)
[7]: (https://www.zhihu.com/question/26419030/answer/274472266)
[8]: (https://www.zhihu.com/question/20587681)
[9]: (ttps://blog.csdn.net/u011508640/article/details/72815981)