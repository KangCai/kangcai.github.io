---
layout: post
title: "C++ 虚继承"
subtitle: "C++ Virtual Inheritance"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 语言基础
---

> 算法。文章首发于[我的博客](https://kangcai.github.io/2018/10/25/ml-overall-bayes/)，转载请保留链接 ;)

1. 通过 “/d1 reportAllClassLayout” 指令查看类内存布局

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c0.PNG"/>
</center>

2. 普通继承

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c1.PNG"/>
</center>

3. 虚继承

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c2.PNG"/>
</center>

其中基类 A 与普通继承一样，故省略；B 由于与 C 是一样的，故也省略。