---
layout: post
title: "机器学习 · 总览篇 XI"
subtitle: "特征工程"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·总览篇
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

### 一、特征工程是什么

### 二、特征构建

### 三、特征清洗

##### 3.1 清洗异常样本

##### 3.2 采样

**样本均衡**

**样本权重**

### 四、单特征处理

不仅仅是对单个特征的处理，也包含对多特征中的某一维特征的处理。

##### 4.1 归一化

应用场景 - 归一化
``1``

##### 4.2 二值化

对于某些应用场景，我们更在乎是与否，而不关注程度，这种时候将定量特征的连续值转换成01值能提高学习效率。定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0（相反也可），公式表达如下：

<center><img src="https://latex.codecogs.com/gif.latex?x'=\left\{\begin{matrix}&space;1,&space;\&space;x>threshold\\&space;0,\&space;x\leq&space;threshold&space;\end{matrix}\right."/></center>

应用场景 - 二值化
``对于一个学生表现相关的分类任务，我们在音乐学习成绩这一项上只关心“及格”还是“不及格”，那么需要将定量的考分，转成“1”和“0”，分别表示及格和不及格，及格阈值是60分，那么大于等于60分转换成1，小于60分转换成0``

##### 4.3 虚拟编码（定性特征表示）

##### 4.4 缺失值

使用不完整的数据集的一个基本策略就是舍弃掉整行或整列包含缺失值的数据。但是这样就付出了舍弃可能有价值数据（即使是不完整的 ）的代价。处理缺失数值的一个更好的策略就是从已有的数据推断出缺失的数值，比如使用缺失数值所在行或列的均值、中位数、众数、最值等来替代缺失值。

应用场景 - 缺失值
````

### 五、多特征处理

##### 5.1 降维

**PCA**

**LDA**

**深度学习**

##### 5.2 特征选择

分为：过滤式（Filter）、封装式（Wrapper）以及 嵌入式（Embedded）。
**过滤式（Filter）**

**封装式（Wrapper）**

**嵌入式（Embedded）**

1. 正则化

2. 决策树

3. 深度学习

[wiki: Normalization (statistics)](https://en.wikipedia.org/wiki/Normalization_(statistics))
[zhihu: 标准化和归一化什么区别？](https://www.zhihu.com/question/20467170)
[cnblogs: 使用sklearn做单机特征工程](http://www.cnblogs.com/jasonfreak/p/5448385.html)
[csdn: 机器学习算法在什么情况下需要归一化](https://blog.csdn.net/sinat_29508201/article/details/53056843)
[jianshu: 归一化、标准化和中心化/零均值化](https://www.jianshu.com/p/95a8f035c86c)