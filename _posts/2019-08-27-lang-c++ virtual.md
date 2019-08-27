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

通过 “/d1 reportAllClassLayout” 指令查看类内存布局**

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c0.PNG"/>
</center>

### 虚继承

虚继承主要用于菱形形式的继承形式，是为了在多继承的时候避免引发歧义，避免重复拷贝。

**1.普通继承**

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c1.PNG"/>
</center>

**2.虚继承**

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c2.PNG"/>
</center>

其中基类 A 与普通继承一样，故省略；B 由于与 C 是一样的，故也省略。以类 C 为例，虚继承的子类中有个虚指针，虚指针指向对应的虚表，虚表中记录了到虚继承基类的偏移。甚至还可以做以下两个操作，

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c3.PNG"/>
</center>

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c4.PNG"/>
</center>

理解了类内存结构、虚指针、虚表的含义，就万变不离其宗了。

### 虚函数

虚函数继承是解决多态性的，当用基类指针指向派生类对象的时候，基类指针调用虚函数的时候会自动调用派生类的虚函数，这就是多态性，也叫动态编联

**1.虚函数**
<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c5.PNG"/>
</center>

**2.虚继承虚函数**

单纯为的是看看类内存的结构

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c6.PNG"/>
</center>

**纯虚函数**

纯虚函数相当于基类只提供接口而不定义具体实现，在函数声明后加=0，如：
virtual void Eat() = 0;