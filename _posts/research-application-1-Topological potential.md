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
4. 角度关系 A。

势能具体衡量的是个体安全程度，安全程度越高，势能越大。

**节点间距离加成 D**

其中，第一项，节点间距离函数图像如下，

<center>
<img src="https://kangcai.github.io/img/in-post/post-research-application/1.PNG" />
</center>

```buildoutcfg
Yd = e^(-(D/d)^2)
```

其中 3d / 2^(0.5) 约等于极限范围，d 为 1 时，极限范围是 2.12。

**本身强度 P**

```buildoutcfg
P' = hp_ratio + max((1 - (spell_cd_left_time / 10)), 1)
P = Yd * P'
Yp = p * P
```

其中 p 是范围在 (0, 1] 之间的常量。

**角度关系加成 A**

```buildoutcfg
Ya = a * P1 * P2 * abs(sin(A / 2))
```

其中 a 是范围在 (0, 1] 之间的常量。

最终总势能为

```buildoutcfg
Y = sum(i)(Yp) + sum(i, j)(Ya)
```

**全局视角**

个人视角计算出的结果表明当前做出改变的危险程度，如果过于危险，就找到附近障碍物躲避；否则参与全局最优计算。

待定方案

> 全局视角根据个人视角的情况，计算全局最优解，找到最近 3 个点。3 x 3 x 3 = 27 种组合中选出最优。其中最为关键的，最优解的标准是全局拓扑势。

最简单方案

> 直接根据当前个人视角的情况，按规则选择出最佳横排战线：tendency 较大时，可三种移动；tendency 较小时，不动。当有两个及两个以上 tendency 较小时，选择最靠后的。

**位移使用方案**

1. tendency 较小 + 转移
2. tendency 较大 + 击杀


