---
layout: post
title: "计算机网络"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 面试
  - 计算机网络
---

> 算法。文章首发于[我的博客](https://kangcai.github.io)，转载请保留链接 ;)


**GET 和 POST 的区别**

一般答案

1. GET在浏览器回退时是无害的，而POST会再次提交请求。 

2. GET产生的URL地址可以被Bookmark，而POST不可以。

3. GET请求会被浏览器主动cache，而POST不会，除非手动设置。

4. GET请求只能进行url编码，而POST支持多种编码方式。

5. GET请求参数会被完整保留在浏览器历史记录里，而POST中的参数不会被保留。

6. GET请求在URL中传送的参数是有长度限制的，而POST么有。

7. 对参数的数据类型，GET只接受ASCII字符，而POST没有限制。

8. GET比POST更不安全，因为参数直接暴露在URL上，所以不能用来传递敏感信息。

HTTP只是个行为准则，而TCP才是GET和POST怎么实现的基本。“一般答案” 里关于参数大小的限制又是从哪来的呢？这是浏览器根据 HTPP 行为准则来做的处理，不同浏览器不一样。

**GET 或 POST 产生多少个 TCP 包**

一般情况下 1 个，但也有 2 个的情况，理论上多少个都行，反正 TCP 协议是流协议。

大多数框架都是尽量在一个tcp包里面把HTTP请求发出去的，但是也确实存在先发HTTP头，然后发body的框架。但是具体发多少个TCP包，这个是代码的问题，是tcp协议栈的问题，跟HTTP没关系。

**HTTP 与 TCP 的联系**

HTTP协议是建立在TCP协议之上的一种应用。

HTTP连接最显著的特点是客户端发送的每次请求都需要服务器回送响应，在请求结束后，会主动释放连接。从建立连接到关闭连接的过程称为“一次连接”。

由于HTTP在每次请求结束后都会主动释放连接，因此HTTP连接是一种“短连接”，要保持客户端程序的在线状态，需要不断地向服务器发起连接请求。通常的 做法是即时不需要获得任何数据，客户端也保持每隔一段固定的时间向服务器发送一次“保持连接”的请求，服务器在收到该请求后对客户端进行回复，表明知道客 户端“在线”。若服务器长时间无法收到客户端的请求，则认为客户端“下线”，若客户端长时间无法收到服务器的回复，则认为网络已经断开。



**参考资料**

1. [GET 和 POST 两种基本请求方法的区别](https://mp.weixin.qq.com/s?__biz=MzI3NzIzMzg3Mw==&mid=100000054&idx=1&sn=71f6c214f3833d9ca20b9f7dcd9d33e4#rd)
2. [http post请求发两个tcp包后续](https://blog.csdn.net/zerooffdate/article/details/81513717)

