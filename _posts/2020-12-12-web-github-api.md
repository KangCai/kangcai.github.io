---
layout: post
title: "【Web】Github API 开发"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - Web

---

**Github API 如何获取某个文件的信息**



---

**Github API 的 Auth 登录，一定要用 get 方法，而不是 post!**

--

**Github API 的 url 里不能直接包含特殊字符**

|       |   URL 中含义  | 十六进制值 |
| :---: | :---: | :---:  |
| + |  +号表示空格  | %2B | 
| 空格 |  URL中的空格可以用+号或者编码  | %20 | 
| / |  分隔目录和子目录  | %2F | 
| ? |  分隔实际的 URL 和参数  | %3F | 
| % |  指定特殊字符  | %25 | 
| # |  表示书签  | %23 | 
| & |  指定的参数间的分隔符  | %26 | 
| = |  指定参数的值 | %3D | 

