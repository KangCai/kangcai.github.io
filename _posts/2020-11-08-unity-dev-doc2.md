---
layout: post
title: "Unity 错误集锦"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - Unity

**多在 [https://docs.microsoft.com/en-us/dotnet/](https://docs.microsoft.com/en-us/dotnet/) 里找答案**

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
