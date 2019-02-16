---
layout: post
title: "机器学习 · 监督学习篇 II"
subtitle: "朴素贝叶斯"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

文本相关分类，比如过滤恶意留言，或者过滤垃圾邮件等任务，一种十分常用的方法就是朴素贝叶斯分类模型。本文将会从概念、公式推导、代码实现、应用等角度详细介绍朴素贝叶斯模型。

贝叶斯很多人都知道或者有所耳闻，统计学上有一个出名的贝叶斯学派和贝叶斯定理：贝叶斯学派详细介绍可参见《总览篇·III》；而贝叶斯定理是上学时可能会接触到的，对于事件 A、B，其发生概率为 P(A)、P(B)，则有，

<img src="https://latex.codecogs.com/gif.latex?P(B|A)=\frac{P(A|B)P(B)}{P(A)}" />

这个定理最大的作用就是告诉我们 A、B 两个事件之间后验概率的关系和转化方式。该定理在机器学习中应用最广泛的两种模型是决策树模型（Decision Tree）和朴素贝叶斯模型（Naive Bayesian Model，NBM）。其中决策树模型是在80年代提出的，而朴素贝叶斯模型发源于古典数学理论有着坚实的数学基础。朴素贝叶斯，之所以称之为朴素，是因为其中引入了几个假设，而正因为这几个假设的引入，使得模型简单、朴素、易理解。

### 一、公式推导

首先，在朴素贝叶斯分类模型中，命，

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;(1)\&space;x=\{a_1,a_2,...,a_m\}\\&space;(2)\&space;C=\{y_1,y_2,...,y_n\}\\&space;\end{aligned}"  />

其中（1）式中 x 是待分类样本，每个 a 是 x 的一个特征属性，（2）式中 C 是类别集合。分类目的是在样本出现 x 特征情况时，该样本最有可能是属于哪一类，朴素贝叶斯做法很直接，先计算该样本属于每个类别的后验概率，

<img src="https://latex.codecogs.com/gif.latex?(3)\&space;P(y_1|x),P(y_2|x),...,P(y_n|x)" />

求得（3）式中样本属于各个类别的后验概率后，选择后验概率最大的类别作为样本分类类别。所以现在的关键点就是如何计算（3）式中的各个条件概率，这个时候就用上了贝叶斯定理，

<img src="https://latex.codecogs.com/gif.latex?(4)\&space;P(y_i|x)=\frac{P(x|y_i)P(y_i)}{P(x)}" />

其中 P(x) 表示的是样本呈现出 x 的特征形式的概率，这个概率通常会看成一个常数，所以我们的目的是求（4）式中分子最大对应的类别 i，而分子都是可以直接求出的量，但问题是 x 本身在训练参考样本中出现的次数就不会太多，各个类别的 <img src="https://latex.codecogs.com/gif.latex?P(x|y_i)"/> 出现的次数就更少了，这样得到的概率值意义不大，所以直接求这个后验概率没什么用。但我们发现 x 的每个属性 a 出现的次数会多很多，所以这里为了让分类模型能够 work，就要引入朴素贝叶斯假设了，即认为各个特征属性 a 都是互相独立的，那么将（4）式分子部分中 <img src="https://latex.codecogs.com/gif.latex?P(x|y_i)"/> 展开有，

<img src="https://latex.codecogs.com/gif.latex?(5)\&space;P(x|y_i)P(y_i)=P(a_1|y_i)P(a_2|y_i)...P(a_m|y_i)=P(y_i)\prod_{j=1}^{m}P(a_j|y_i)"/>

所以最后朴素贝叶斯模型要做的事情就只是统计一下各个类别 y 下的各个 a 出现的概率，以及类别 y 本身出现的概率，就可以对新样本 x' 进行分类了。

从上述公式推导过程中可以看到，朴素贝叶斯不同于其它模型，没有复杂的优化过程，所需估计的参数也很少，效率极高，对缺失数据不太敏感，理论上，朴素贝叶斯分类模型与其他分类方法相比具有最小的误差率；但由于各个特征属性 a 都是互相独立的这一朴素假设在实际应用中往往是不成立的，朴素贝叶斯分类效果并没有理论上的那么完美。

### 二、Python 实现

如文章开头所述，一个典型的朴素贝叶斯模型应用场景是垃圾文本过滤，本小节就只用 numpy 库手写实现手机短信垃圾（SMS Spam）分类，选用的数据集是经典的 [SMS Spam Collection v. 1](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)，共5,574条短信，其中垃圾短信747条，非垃圾短信4827条。

##### 2.1 手写实现

首先是不使用机器学习库的手写实现，主要包括 特征提取、训练、预测 三个过程。

**2.1.1 特征提取** 

“SMS Spam Collection v. 1” 数据集格式如下所示，

```buildoutcfg
1. ham   MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*
2. ham   Siva is in hostel aha:-.
3. spam  FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop
4. ham   Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.
5. spam  Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B
```

每个样本一行，上面的示例一共有5个样本。对于1、2、4属于正样本，即正常信息，故标签是"ham"；3、5属于负样本，即广告等垃圾消息，故标签是"spam"。标签后面就是具体的消息内容。

特征提取的第一步是将句子切分成单词，由于是英文，所以这里处理方式比较简单暴力，按照空格和除'之外的符号来切分了，然后全部转小写。为了选出可以作为特征的单词，可以统计单词出现的文档次数，并试图把直观上无效（出现在的文档数目较少）的单词排除掉，当然了，也可以把所有单词都作为特征，这里只是想尝试一下。统计结果如下图X所示，

<center>
<img src="https://kangcai.github.io/img/in-post/post-ml/word_statistic.png"/>
</center>
<center>图1 单词出现频次统计（前10%）</center>

共有7956种单词出现，图1表明了绝大部分单词出现频次都相当低。后文中，我们分别选取了前200、500、2000、5000、7956这5种截断方式来作为特征，特征维度对应的就是200、500、2000、5000、7956维。

**2.1.2 训练** 

朴素贝叶斯没有显式的训练过程，所谓的训练过程只是计算：样本类别在样本中出现的先验概率 和 关键字|类别 的条件概率。以上的参数就是模型的全部参数，也是可以直接从训练样本计算得到的，没有最优化过程。下面是相当朴素的 python 实现代码，本质上就是在计算 P(类别) 和 P(关键字|类别)。

```buildoutcfg
class NaiveBayesClassificationModel(object):
    """
    朴素贝叶斯模型
    """
    def __init__(self, kw_set):
        # 关键字集合，即哪些单词是我们要当做是特征属性的单词
        self.kw_set = kw_set
        # P(类别) 样本类别本身在样本中出现的先验概率
        self.label_prior_prob = dict()
        # P(关键字|类别) 这一条件概率
        self.kw_posterior_prob = dict()

    def train(self, data):
        """
        训练模型
        :param data: 以 [[label] [input_text_words]] 的形式构成的list
        :return: None
        """
        # 计算条件概率 P(关键字|类别)
        for label, input_text_words in data:
            if label not in self.kw_posterior_prob:
                self.kw_posterior_prob[label] = dict()
            if label not in self.label_prior_prob:
                self.label_prior_prob[label] = 0
            self.label_prior_prob[label] += 1
            for word in input_text_words:
                if word not in self.kw_set:
                    continue
                if word not in self.kw_posterior_prob[label]:
                    self.kw_posterior_prob[label][word] = 0
                self.kw_posterior_prob[label][word] += 1
        for label, kw_posterior_prob in self.kw_posterior_prob.items():
            for word in self.kw_set:
                if word in kw_posterior_prob:
                    self.kw_posterior_prob[label][word] /= self.label_prior_prob[label] * 1.0
                else:
                    self.kw_posterior_prob[label][word] = 0
        # 样本类别本身在样本中出现的先验概率 P(类别)
        for label in self.label_prior_prob:
            self.label_prior_prob[label] /= len(data) * 1.0
```

**2.2.3 预测**

预测过程也相当简单，实现这个式子即可，

<img src="https://latex.codecogs.com/gif.latex?(5)\&space;P(x|y_i)P(y_i)=P(a_1|y_i)P(a_2|y_i)...P(a_m|y_i)=P(y_i)\prod_{j=1}^{m}P(a_j|y_i)"/>

下面就是简单的 python 代码实现，

```buildoutcfg
    def predict(self, input_text):
        """
        预测过程
        :param input_text: 处理过后的单词集合
        :return:
        """
        predicted_label = None
        max_prob = None
        for label in self.label_prior_prob:
            prob = 1.0
            for word in self.kw_set:
                if word in input_text:
                    prob *= self.kw_posterior_prob[label][word]
                else:
                    prob *= 1 - self.kw_posterior_prob[label][word]
            if max_prob is None or prob > max_prob:
                predicted_label = label
                max_prob = prob
        return predicted_label
```

完整代码可见 github： [https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/code/nb.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/code/nb.py)。

##### 2.2 scikit-learn 实现

上一小节 2.1 中采用的实现方式是：某个单词在文档中出现过，则其特征值为1，否则为0，所以采用的是 BernoulliNB 伯努利模型。scikit-learn 具体实现如下

```buildoutcfg
from sklearn.naive_bayes import BernoulliNB
# 针对本例子，使用 BernoulliNB Naive Bayes；
bnb = BernoulliNB()
bnb.fit(X, Y)
result_predict = bnb.predict(X')
```

与伯努利模型一样，多项式模型 MultinomialNB 同样是适用于离散特征的情况， 多项式模型，差别在于，计算条件概率时，多项式模型的特征取值不是1或0，还需要考虑样本个数。除此之外，skikit-learn 还有提供解决特征是连续变量的 GaussianNB 高斯模型，高斯模型与多项式模型唯一不同的地方就在于，计算条件概率时，高斯模型假设各维特征服从正态分布，需要计算的是各维特征的均值与方差。

##### 2.3 交叉验证结果

通过2.1中的代码实现，做4折-交叉验证，准备率为

|  | 200 | 500 | 2000 | 5000 | 7956 |
| :-----------:| :----------: |:----------: | :----------: | :----------: | :----------: | 
| 准确率 | 97.9%  | 99.28% | 99.83% | 99.95% | 99.93% |
| 时间(秒) | 0.27  | 0.63 | 2.48 | 4.67 | 4.69 |

从上表可以看到，维度越高，准确率基本是呈提升趋势的，当然耗时也会随之线性增加。但需要注意一点，即使在本实验中采取的是交叉验证方式，这也只能高维度特征的朴素贝叶斯模型针对本数据集是有效的，对于新数据，不一定具有很强的泛化能力。所以我认为选取200或者500维度的高频词单词作为特征，虽然准确率稍微低一点，但具备更强的泛化能力。当然，如果数据来源确实是真实来源，且足够多（本实验数据只有5574条短信），那么实验环境下的高准确率还是具备很强的指导作用的。

另外，为了进一步地研究朴素贝叶斯模型针对垃圾消息分类任务的适用性，下面列出了当维度为5000时的混淆矩阵，

|  | 判成正常 | 判成垃圾 |
| :-----------:| :----------: |:----------: |
| 正常 | 100% | 0.5% |
| 垃圾 | 0% | 99.5% |

可以看到正常消息判成正常的概率是100%，而垃圾消息会有极小部分会判成正常，这种混淆矩阵情况对于实际应用是相当合适的，因为用户必然不会接受出现正常消息被阻挡的情况发生，而极个别垃圾消息没有被阻挡掉在一定程度是可以理解的。

综合以上实验，我们发现即使是对于出现频次相当低的单词特征，朴素贝叶斯模型也能有效地利用，提高模型的预测准确率；而且朴素贝叶斯模型的参数没有最优化过程，参数是直接获取的，这使得模型创建过程简单、可解释性强。相比于其它模型，朴素贝叶斯模型在垃圾（也包括政治敏感、赌博、色情、违法犯罪等）本文分类任务上有很大的优势。

1. [wiki: 朴素贝叶斯分类器](https://zh.wikipedia.org/wiki/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%88%86%E7%B1%BB%E5%99%A8)
2. [csdn: 朴素贝叶斯理论推导与三种常见模型](https://blog.csdn.net/u012162613/article/details/48323777)
3. [csdn: python_sklearn机器学习算法系列之sklearn.naive_bayes朴树贝叶斯算法](https://blog.csdn.net/weixin_42001089/article/details/79952245)