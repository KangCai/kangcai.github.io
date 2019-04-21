---
layout: post
title: "机器学习 · 监督学习篇 IV"
subtitle: "支持向量机"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

核函数映射
https://www.cnblogs.com/machinelearner/archive/2012/12/31/2841175.html

SMO算法
http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html


### 二、LR 的 Python 实现

还是使用垃圾信息分类任务为例，选用的数据集是经典的 [SMS Spam Collection v. 1](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)，共5,574条短信，其中垃圾短信747条，非垃圾短信4827条。“SMS Spam Collection v. 1” 数据集格式如下所示，

```buildoutcfg
1. ham   MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*
2. ham   Siva is in hostel aha:-.
3. spam  FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop
4. ham   Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.
5. spam  Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B
```

每个样本一行，上面的示例一共有5个样本。1、2、4属于正样本，即正常消息，标签是"ham"；3、5属于负样本，即广告等垃圾消息，标签是"spam"。标签后面就是具体的消息内容。

##### 2.1 不使用机器学习库的 numpy 实现

本小节就只用 numpy 库手写实现手机短信垃圾（SMS Spam）分类。

**1.2.1 特征提取**

特征提取的第一步是将句子切分成单词，由于是英文，所以这里处理方式比较简单暴力，按照空格和除'之外的符号来切分了，然后全部转小写。用热编码特征，来表示单词是否出现，出现的单词对应特征维度值为1，未出现对应值为0。

```buildoutcfg
def feature_batch_extraction(d_list, kw_set):
    """
    特征批量提取
    :param d_list: 原始数据集
    :param kw_set: 关键字列表
    :return:
    """
    kw_2_idx_dict = dict(zip(list(kw_set), range(len(kw_set))))
    feature_data = np.zeros((len(d_list), len(kw_set)))
    label_data = np.zeros((len(d_list), 1))
    for i in range(len(d_list)):
        label, words = d_list[i]
        for word in words:
            if word in kw_2_idx_dict:
                feature_data[i, kw_2_idx_dict[word]] = 1
        label_data[i] = 1 if label == 'spam' else 0
    return feature_data, label_data
```
**1.2.2 模型**

```buildoutcfg

```

**1.2.3 训练**

训练的过程就是将1.1 节的公式用代码实现一遍，

```buildoutcfg

```

**1.2.4 验证**

用准确率和混淆矩阵两个指标来进行验证，

```buildoutcfg

```

##### 2.2 scikit-learn 实现


```buildoutcfg
from sklearn import svm
svm_model = svm.SVC()
lr.fit(X, Y)
result_predict = lr.predict(X')
```