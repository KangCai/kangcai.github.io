---
layout: post
title: "【Web】服务端"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - Web

---

**Django部署**

Django 管理工具 django-admin.py

使用 django-admin.py 来创建 HelloWorld 项目：`django-admin.py startproject HelloWorld`

接下来我们进入 HelloWorld 目录输入以下命令，启动服务器：`python3 manage.py runserver 0.0.0.0:8000`

[报错：SQLite 3.8.3 or later is required (found 3.7.17)](https://blog.csdn.net/qq_39969226/article/details/92218635)

[报错：Invalid HTTP_HOST header: 'xxx.xx.xxx.xxx:8000'. You may need to add 'xxx.xx' to ALLOWED_HOSTS！](https://blog.csdn.net/lezeqe/article/details/83820621)

Django 工程相关知识：

1. `BASE_DIR` 是在文件夹工程中与 manager.py 同级的目录；
2. path 可以指定访问的 url
3. setting.py 里可以设置 BASE_DIR、模板目录；
4. 模块 html 中，双括号 `{{X}}` 表示变量 X，通过 `render(request, 'runoob.html', context)` 中的 context['X'] 赋值可以传进来，其中 request 是 path 或 url 调用函数默认参数；
5. 除了上述模板外，最简单的方式就是 `HttpResponse("Hello world ! ")`
6. 模板中对象除了可以传文本外，链接等都可以传。
7. `{% include %}` 标签允许在模板中包含其它的模板的内容，比如下面这个例子包含了 `nav.html` 模板：·`{% include "nav.html" %}`
8. 在开发机本地离线模拟好逻辑，再移植到同步模式更好

**Django css等静态资源在 Debug 模式加载不成功等问题**

[《在Django中加载css实例》](https://blog.csdn.net/Pansc2004/article/details/80553573)

--

