---
layout: post
title: "面试 · C++指针"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 面试
  - 语言基础
  - C++
---

> 算法。文章首发于[我的博客](https://kangcai.github.io)，转载请保留链接 ;)


### dynamic_cast

来源于 **RTTI（Run-Time Type Identification)**，通过运行时类型信息程序能够使用基类的指针或引用来检查这些指针或引用所指的对象的实际派生类型。

将一个基类对象指针（或引用）cast到继承类指针，dynamic_cast会根据基类指针是否真正指向继承类指针来做相应处理：具体来说会做类型检查，如果类型对了，才返回转换结果；类型不对，返回空指针。

"dynamic_cast" 用于在类层次结构中漫游，对指针或引用进行自由的向上、向下或交叉强制。"typeid" 则用于获取一个对象或引用的确切类型，与 "dynamic_cast" 不同，将 "typeid" 作用于指针通常是一个错误，要得到一个指针指向之对象的type_info，应当先将其解引用（例如："typeid(*p);"）。

一般地讲，能用虚函数解决的问题就不要用 "dynamic_cast"，能够用 "dynamic_cast" 解决的就不要用 "typeid"。比如：

```buildoutcfg
//反面例子
void
rotate(IN const CShape& iS)
{
    if (typeid(iS) == typeid(CCircle))
    {
        // ...
    }
    else if (typeid(iS) == typeid(CTriangle))
    {
        // ...
    }
    else if (typeid(iS) == typeid(CSqucre))
    {
        // ...
    }

    // ...
}
```

**以上代码用 "dynamic_cast" 写会稍好一点，当然最好的方式还是在CShape里定义名为 "rotate" 的虚函数。** dynamic_cast 基本上用于把 Base 类指针转换为 Derived 类指针 （即 downcast）。如果程序的运行逻辑能保证给定的 downcast 一定合法，除非继承中有用 virtual 继承，否则 dynamic_cast 可以用 static_cast 替代，这样没有运行时开销。设计合理的程序通常能保证 downcast 一定合法。例如最常见的 Foo* -> FooImpl*，这里 Foo 是接口类 （pure interface）。

> dynamic_cast 是有可能抛出 std::bad_cast 异常的 

**只有对引用做 dynamic_cast 时才会抛出 std::bad_cast 异常，对指针做 dynamic_cast 失败是通过返回 null 来通知调用者的（意即：对指针做 dynamic_cast 时不会抛出任何异常）。再说 dynamic_cast 本就不该被频繁使用，更别提将其用在引用而不是指针上了（反正我至今也未在真实生产代码中对引用做过 dynamic_cast）。另外：合理使用异常也没什么不好。**


### shared_ptr

C语言、C++语言没有自动内存回收机制，程序员每次new出来的内存块都需要自己使用delete进行释放，流程复杂可能会导致忘记释放内存而造成内存泄漏。而智能指针也致力于解决这种问题，使程序员专注于指针的使用而把内存管理交给智能指针。

我们先来看看普通指针的悬垂指针问题。当有多个指针指向同一个基础对象时，如果某个指针delete了该基础对象，对这个指针来说它是明确了它所指的对象被释放掉了，所以它不会再对所指对象进行操作，但是对于剩下的其他指针来说呢？它们还傻傻地指向已经被删除的基础对象并随时准备对它进行操作。

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c10.PNG"/>
</center>

通过计数来解决该问题，当只有一个指针指向基础对象的时候，这时通过该指针就可以大大方方地把基础对象删除了。share_ptr 就是这种计数的智能指针。

```buildoutcfg
#include "stdafx.h"
#include <iostream>
#include <memory>

using namespace std;
class A
{
public:
	int i;
	A(int n) :i(n) { };
	~A() { cout << i << " " << "destructed" << endl; }
};
int main()
{
	shared_ptr<A> sp1(new A(2)); //A(2)由sp1托管，
	shared_ptr<A> sp2(sp1);       //A(2)同时交由sp2托管
	shared_ptr<A> sp3;
	sp3 = sp2;   //A(2)同时交由sp3托管
	cout << sp1->i << "," << sp2->i << "," << sp3->i << endl;
	A * p = sp3.get();      // get返回托管的指针，p 指向 A(2)
	cout << p->i << endl;  //输出 2
	sp1.reset(new A(3));    // reset导致托管新的指针, 此时sp1托管A(3)
	sp2.reset(new A(4));    // sp2托管A(4)
	cout << sp1->i << endl; //输出 3
	sp3.reset(new A(5));    // sp3托管A(5),A(2)无人托管，被delete
	cout << "end" << endl;
	system("pause");
	return 0;
}
```

控制台输出的结果如下，

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c11.PNG"/>
</center>


**1.手动实现 share_ptr**

```buildoutcfg
#include "stdafx.h"
#include <iostream>
using namespace std;

class Point
{
private:
	int x, y;
public:
	Point(int xVal = 0, int yVal = 0) :x(xVal), y(yVal) { }
	int getX() const { return x; }
	int getY() const { return y; }
	void setX(int xVal) { x = xVal; }
	void setY(int yVal) { y = yVal; }
};

class U_Ptr
{
private:
	friend class SmartPtr;
	U_Ptr(Point *ptr) :p(ptr), count(1) { }
	~U_Ptr() { delete p; }

	int count;
	Point *p;
};

class SmartPtr
{
public:
	SmartPtr(Point *ptr) :rp(new U_Ptr(ptr)) { }

	SmartPtr(const SmartPtr &sp) :rp(sp.rp) { ++rp->count; }

	SmartPtr& operator=(const SmartPtr& rhs) {
		++rhs.rp->count;
		if (--rp->count == 0)
			delete rp;
		rp = rhs.rp;
		return *this;
	}

	~SmartPtr() {
		if (--rp->count == 0)
			delete rp;
		else
			cout << "还有" << rp->count << "个指针指向基础对象" << endl;
	}

private:
	U_Ptr *rp;
};

int main()
{
	//定义一个基础对象类指针
	Point *pa = new Point(10, 20);

	//定义三个智能指针类对象，对象都指向基础类对象pa
	//使用花括号控制三个指针指针的生命期，观察计数的变化

	{
		SmartPtr sptr1(pa);//此时计数count=1
		{
			SmartPtr sptr2(sptr1); //调用复制构造函数，此时计数为count=2
			{
				SmartPtr sptr3 = sptr1; //调用赋值操作符，此时计数为conut=3
			}
			//此时count=2
		}
		//此时count=1；
	}
	//此时count=0；pa对象被delete掉

	cout << pa->getX() << endl;

	system("pause");
	return 0;
}
```

控制台输出的结果如下，

<center>
<img src="https://kangcai.github.io/img/in-post/post-lang/c12.PNG"/>
</center>



**参考资料**

1.[RTTI、虚函数和虚基类的实现方式、开销分析及使用指导](http://baiy.cn/doc/cpp/inside_rtti.htm)
2.[为什么说不要使用 dynamic_cast，需要运行时确定类型信息，说明设计有缺陷？](https://www.zhihu.com/question/22445339)
3.[shared_ptr基于引用计数智能指针实现](https://blog.csdn.net/qq_16209077/article/details/52791434)