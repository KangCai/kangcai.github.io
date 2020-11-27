---
layout: post
title: "【游戏框架】2-关于ECS的思考"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - GamePlay
---

> 算法。文章首发于[我的博客](https://kangcai.github.io)，转载请保留链接 ;)


**对于非单例 component，在何时、何处进行创建和绑定**

感觉这个还是要根据具体情形进行设计，但至少得有一点，

创建玩家：在游戏开局初始化时、；