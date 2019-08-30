---
layout: post
title: "Tensorflow I Estimator 使用+坑"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - Tensorflow
---

> 算法。文章首发于[我的博客](https://kangcai.github.io)，转载请保留链接 ;)


**坑一：Estimator 每 predict 一次都要重新加载一次模型**

**问题描述**：

使用了一个宽深神经网络模型，GTX 2070 下每调用 predict 平均耗时 300+ ms（主要用来加载模型），这根本没法直接投入到产品中，没有其他接口可以调用。

**解决方案**：

网上给出了通过生成器的参数输入的方式让模型只加载一次，使用后 GTX 2070 每次 predict 平均耗时 1.5 ms，很稳；然后由于只 predict 一个样本，用 CPU 更快，平均 0.5 ms。主要参考帖子：[https://github.com/tensorflow/tensorflow/issues/4648](https://github.com/tensorflow/tensorflow/issues/4648)

```buildoutcfg
class FastPredict:

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn

    def _create_generator(self):
        while not self.closed:
            yield self.next_features

    def predict(self, feature_batch):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.
            feature_batch a list of list of features. IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list (i.e. predict([my_feature]), not predict(my_feature)
        """
        self.next_features = feature_batch
        if self.first_run:
            self.batch_size = len(feature_batch)
            self.predictions = self.estimator.predict(input_fn=self.input_fn(self._create_generator))
            self.first_run = False
        elif self.batch_size != len(feature_batch):
            raise ValueError("All batches must be of the same size. First-batch:" + str(self.batch_size) + " This-batch:" + str(len(feature_batch)))

        return next(self.predictions)

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except:
            print("Exception in fast_predict. This is probably OK")
```

参考代码链接：[https://github.com/marcsto/rl/blob/master/src/fast_predict2.py](https://github.com/marcsto/rl/blob/master/src/fast_predict2.py)，链接中代码有个笔误的 bug。另外需要注意的是，**在使用 FastPredcit 对象的 predict 功能时，即使是 1 个样本，外面也要多套一层：predict(\[my_feature\]（正确），predict(my_feature)（错误）。**