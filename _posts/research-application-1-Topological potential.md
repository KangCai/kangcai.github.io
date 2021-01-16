---
layout: post
title: "【研究应用】1-拓扑势"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 研究应用
---

多人合作对抗游戏中，各个 agent 形成了典型拓扑结构，每个 agent 相当于一个节点，在 AI 开发中，我们尝试有效利用节点间的拓扑关系。

## 设计方法

根据具体玩法，设计合理方法，对拓扑关系加以利用。首先是拓扑势

**拓扑势**

拓扑势（Topological potential）概念常应用于网络社区相关势场。

<center>
<img src="https://latex.codecogs.com/gif.latex?\varphi(v&space;i)=\sum_{j=1}^{n}\left(m&space;j&space;\times&space;e^{-\left(\frac{d&space;i&space;j}{\sigma}\right)^{2}}\right)"/>
</center>

实际应用场景中，各项参数与对应的定性关系如下：

1. 节点间距离 D。负相关。
2. 节点本身强度 P。正相关。
3. 节点阵营 F。同阵营加成，反阵营削弱。
4. 角度关系。

势能具体衡量的是个体安全程度，安全程度越高，势能越大。

其中，第一项，节点间距离函数图像如下，

<center>
<img src="https://kangcai.github.io/img/in-post/post-research-application/1.PNG" />
</center>