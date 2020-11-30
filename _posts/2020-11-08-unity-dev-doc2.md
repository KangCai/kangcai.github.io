---
layout: post
title: "【Unity】2-报错集锦"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - Unity

--

**多在 [https://docs.microsoft.com/en-us/dotnet/](https://docs.microsoft.com/en-us/dotnet/) 里找答案**

**CS7036: 未提供与“FState<TestState>.FState(TestState)”的必需形参“state”对应的实参**

[https://zhidao.baidu.com/question/1496589616910039099.html](https://zhidao.baidu.com/question/1496589616910039099.html)

---

**CS0019: 运算符“==”无法应用于“T”和“T”类型的操作数**

[C#中泛型类型的比较（运算符==无法用于T和T类型的操作数）:https://blog.csdn.net/birdfly2015/article/details/94609235](https://blog.csdn.net/birdfly2015/article/details/94609235)

---

**StackOverflowException: The requested operation caused a stack overflow. IterComp`1[T].GetEnumerator () <0x1a82dcafbd0 + 0x00008> in <d553ff918ab24d0aaa944212c35416af>:0**

正确写法：

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

**CS0266:无法将类型“CompBase”隐式转换为“T”。存在一个显式转换(是否缺少强制转换?)**

[《C# 泛型 无法将类型xx隐式转换为“T”》：https://blog.csdn.net/wulijian/article/details/43084121](https://blog.csdn.net/wulijian/article/details/43084121)

---

**CS1579:“IEnumerator”不包含“GetEnumerator”的公共实例定义，因此 foreach 语句不能作用于“IEnumerator”类型的变量**

[《IEnumerable与IEnumerator学习（一）：在类中添加GetEnumerator()方法使类或类的集合可以被迭代》：https://blog.csdn.net/cyh1992899/article/details/52782818](https://blog.csdn.net/cyh1992899/article/details/52782818)

---

**CS0052:可访问性不一致:字段类型“EGameMap”的可访问性低于字段“GameLevelSingleton.gameMap”**

`enum EGameMap`**` 改成 `public enum EGameMap`

[https://zhidao.baidu.com/question/214553362.html](https://zhidao.baidu.com/question/214553362.html)

---

**CS0417:"T":创建变量类型的实例时无法提供参数**

[《C＃中的泛型 - 如何使用参数创建变量类型的实例？》:https://www.javaroad.cn/questions/122324](https://www.javaroad.cn/questions/122324)

---

**NullReferenceException: Object reference not set to an instance of an object**

https://docs.microsoft.com/en-us/dotnet/api/system.reflection.propertyinfo.setvalue?view=netcore-3.1#System_Reflection_PropertyInfo_SetValue_System_Object_System_Object_

https://bbs.csdn.net/topics/30043235

https://stackoverflow.com/questions/48407783/c-sharp-propertyinfo-setvalue-and-array

---

**ScriptableObject 导表后结构体不显示**

变量全由 private 改成 public

---

**类型’T’不能用作泛型类型或方法中的类型参数’T’**

将通用约束附加到你的方法,以确保你的 T 是组件

```buildoutcfg
protected void AttachComponent<T>() where T : Component
{
    // Your code.
}
```

[http://www.voidcn.com/article/p-qzrmhfsz-bvn.html](http://www.voidcn.com/article/p-qzrmhfsz-bvn.html)

---

**Setting the parent of a transform which resides in a Prefab Asset is disable**

[https://blog.csdn.net/yaoning6768/article/details/103809531](https://blog.csdn.net/yaoning6768/article/details/103809531)
