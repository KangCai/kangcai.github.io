---
layout: post
title: "机器学习 · 强化学习篇 VI"
subtitle: "Actor Critic"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·强化学习篇
---

结合了 Policy Gradient (Actor) 和 Function Approximation (Critic) 的方法。Actor 基于概率选行为, Critic 基于 Actor 的行为评判行为的得分, Actor 根据 Critic 的评分修改选行为的概率.

Policy-based、Off-policy（？）、

Actor-Critic（AC）

Advantage Actor-Critic（A2C）

Asynchronous Advantage Actor-Critic（A3C）