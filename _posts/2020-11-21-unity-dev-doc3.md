---
layout: post
title: "【Unity】3-C#"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - Unity

---

**C# 注释规范**

[https://www.cnblogs.com/luzhihua55/p/CodingConventions4All.html](https://www.cnblogs.com/luzhihua55/p/CodingConventions4All.html)

不要用注释来粉饰糟糕的代码，写注释常见的动机之一就是试图来使糟糕的代码能让别人看懂。对于这种"拐杖式注释"，我们不需要，我们要做的是把代码改的能够更具有"自我说明性"。记住："好代码>坏代码+好注释"。

---

****

[《测试框架nunit之assertion断言使用详解》:https://www.jb51.net/article/46290.htm](https://www.jb51.net/article/46290.htm)

---

**C# 自定义类的对象判断为 null 的问题**

[《C#中自定义了一个类实例化后，系统判定对象为空》:https://ask.csdn.net/questions/901987](https://ask.csdn.net/questions/901987)

---

**C# 的迭代器**

[《C#中的IEnumerator 和 yield》https://blog.csdn.net/u012138730/article/details/81148086](https://blog.csdn.net/u012138730/article/details/81148086)

示例：
```buildoutcfg
using System;
using System.Collections;
using System.Collections.Generic;

public class IterComp<T>: IEnumerable<T> where T: CompBase
{
    private static readonly Lazy<IterComp<T>> Instancelock = new Lazy<IterComp<T>>(() => new IterComp<T>());

    public static IterComp<T> GetInstance
    {
        get
        {
            return Instancelock.Value;
        }
    }

    public IEnumerator<T> GetEnumerator()
    {
        string typeName = typeof(T).ToString();
        if (!EntityAdmin.GetInstance.compDict.ContainsKey(typeName))
            yield break;
        foreach (int guid in EntityAdmin.GetInstance.compDict[typeName])
        {
            T comp = EntityAdmin.GetInstance.getComp<T>(guid);
            yield return comp;
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
```

---

**C# 的集合**

HashSet\<T\>

---

**C# 的反射**

[《C# 利用类名字符串调用并执行类方法》：https://blog.csdn.net/u014786187/article/details/105912828/](https://blog.csdn.net/u014786187/article/details/105912828/)
