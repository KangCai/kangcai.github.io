---
layout: post
title: "C++ 虚继承与虚函数"
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

虚继承主要用于菱形形式的继承形式，是为了在多继承的时候避免引发歧义，避免重复拷贝。重要概念是 **虚基类指针(vbptr)** 和 **虚基类表(vftable)**。

**1.普通继承**

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c1.PNG"/>
</center>

**2.虚继承**

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c2.PNG"/>
</center>

其中基类 A 与普通继承一样，故省略；B 由于与 C 是一样的，故也省略。以类 C 为例，虚继承的子类中有个虚基类指针，虚基类指针指向对应的虚表，虚基类表中记录了到虚基类的偏移。甚至还可以做以下两个操作，

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c3.PNG"/>
</center>

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c4.PNG"/>
</center>

理解了类内存结构、虚基类指针、虚基类表的含义，就万变不离其宗了。

### 虚函数

虚函数继承是解决多态性的，当用基类指针指向派生类对象的时候，基类指针调用虚函数的时候会自动调用派生类的虚函数，这就是多态性，也叫动态编联。重要概念是 **虚函数指针(vfptr)** 和 **虚函数表(vftable)**

**1.虚函数**
<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c5.PNG"/>
</center>

**2.虚继承虚函数**

单纯为的是看看类内存的结构

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c6.PNG"/>
</center>

**3.纯虚函数**

纯虚函数相当于基类只提供接口而不定义具体实现，在函数声明后加=0，如：
virtual void Eat() = 0。重要概念是 **抽象类**。

在基类中不能对虚函数给出**有意义**的实现，凡是含有纯虚函数的类叫做抽象类。这种类不能声明对象，只是作为基类为派生类服务。除非在派生类中完全实现基类中所有的的纯虚函数，否则，派生类也变成了抽象类，不能实例化对象。

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c7.PNG"/>
</center>

一般而言纯虚函数的函数体是缺省的，但是也可以给出纯虚函数的函数体（此时纯虚函数变为虚函数），这一点经常被人们忽视，调用纯虚函数的方法为baseclass::virtual function。

### Q&A

**1.虚函数（virtual）能是static的吗？**

不能，因为静态成员函数可以不通过对象来调用，即没有隐藏的this指针；而virtual函数一定要通过对象来调用，即有隐藏的this指针

**2.能调用纯虚函数吗？**

能，通过静态调用

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c8.PNG"/>
</center>

**3.抽象类的继承类不实现纯虚函数会怎样？**

那么继承类本身也是抽象类，不能直接用来实例化对象。

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c9.PNG"/>
</center>