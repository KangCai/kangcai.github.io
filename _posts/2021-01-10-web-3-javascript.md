---
layout: post
title: "【Web】服务端"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - Web

---

**js 修改 html 中 DOM 属性**


JQuery 如下

```buildoutcfg
let tabBtnList = $(".tabBtnList");
if (scrollTop >= 66 && tabBtnList.css('position') === 'static') {
    tabBtnList.css('position', 'fixed');
    tabBtnList.css('box-shadow', '0 1px 3px rgba(18,18,18,.1)');
} else if (scrollTop < 66 && tabBtnList.css('position') === 'fixed') {
    tabBtnList.css('position', 'static');
    tabBtnList.css('box-shadow', 'none');
}
```

原生 js 如下

```buildoutcfg
$(".tabBtnList").each(function(index, domEle) {
    if (scrollTop >= 66){
        domEle.setAttribute('style', 'top:0px;position:fixed;box-shadow: 0 1px 3px rgba(18,18,18,.1)');
    } else{
        domEle.setAttribute('style', 'position:relative');
    }
})
```

---

**ajax 异步模式**

```buildoutcfg
$(function(){
    $("#b01").on("click", function(){
        let htmlObj = $.ajax({
            url:"jquery/test1.txt",
            async:true,
            complete: function(msg){
            },
            success : function(data) {
                $("#myDiv").html(data);
            }
        });
    });
});
```

---

**ajax 同步模式（已弃用）**

```buildoutcfg
$(function(){
    $("#b01").on("click", function(){
        let htmlObj = $.ajax({
            url:"jquery/test1.txt",
            async:false
        });
        $("#myDiv").html(htmlObj.responseText);
    });
});
```
---

**javascript 三种字符串拼接方法**

[JS中三种字符串连接方式及其性能比较](https://www.cnblogs.com/programs/p/5554742.html)

```buildoutcfg
str += "a";
```

```buildoutcfg
var arr=new Array();
arr.push("a");
arr.push("b");
var str = arr.join("");
```

```buildoutcfg
str = str.concat("a"); 
```

结论：三种方法性能差不多

---

**webstorm 编辑器中调用 jquery 语法会出现提醒警告问题**

[Configure JavaScript libraries](https://intellij-support.jetbrains.com/hc/en-us/community/posts/360002260719--jQuery-shortcut-underlined-as-unresolved-function-or-method-)