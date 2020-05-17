---
layout: post
title: "机器学习 · 监督学习篇 VIII"
subtitle: "决策树"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 机器学习
  - 机器学习·监督学习篇
---

本质上，树模型拟合出来的函数其实是分区间的阶梯函数。树形模型具有的最大优点是：更加接近人的思维方式，产生的模型具有可解释性，而且可以直接得到可视化的分类规则。

跟之前一样，本文仍采用银行贷款资质判定的这个例子来对决策树进行解释，如下表所示，前 4 列属性，包括 “年龄”、是否“有工作”、是否“有房”、“信贷情况”是否良好，是 4 个维度的特征，银行根据该 4 个特征来判定是否批准贷款。表格中一共有15个样本，表格最后一列是“是否批准贷款”的实际结果，作为训练标签，

|  | 年龄 | 有工作 | 有房 | 信贷情况 | 类别（标签） |
| :-----------:| :----------: |:----------: | :----------: | :----------: | :----------: | 
| 1 | 青年  | 否|否|一般|否|
| 2 | 青年  | 否|否|好|否|
| 3 | 青年  | 是|否|好|是|
| 4 | 青年  | 是|是|一般|是|
| 5 | 青年  | 否|否|一般|否|
| 6 | 中年  | 否|否|一般|否|
| 7 | 中年  | 否|否|好|否|
| 8 | 中年  | 是|是|好|是|
| 9 | 中年  | 否|是|非常好|是|
| 10 | 中年  | 否|是|非常好|是|
| 11 | 老年  | 否|是|非常好|是|
| 12 | 老年  | 否|是|好|是|
| 13 | 老年  | 是|否|好|是|
| 14 | 老年  | 是|否|非常好|是|
| 15 | 老年  | 否|否|一般|否|

针对这个问题，我们用简单的 if else 语句来解决，但直观上我们很难判定这种模型是不是最优模型，那有什么通用标准能帮我们找到一个最优模型呢？下面将针对该问题来引出 3 种经典决策树方法: ID3算法、C45算法、CART算法。

3 种算法主要差别在于对特征选择的标准，其它过程基本一致，都分为以下 5 个步骤吗，对于特征集 A，标签集 D，设定一个空的根节点，将其作为当前节点：

1. 计算特征集 A 中各特征对 D 的信息增益，选择信息增益最大的特征 A_g，存入当前节点；
2. 如果 A_g 的信息增益小于阈值 epsilon，则以当前节点为根节点的决策树就成了一个单节点树，直接返回；
3. 对于 A_g 的每一可能值 a_i，依据 A_g = a_i 将 D 分割为若干非空子集 D_i，将当前结点的标记设为样本数最大的 D_i 对应的类别。遍历每一个 D_i，对其中的各个 Di 都分别以 D_i 为训练集，以 A - {A_g} 为特征集，得到子树 T_i，并将该子树 T_i 作为返回值返回到步骤 1。

这 3 个步骤可用以下 python 代码体现，即 _train 函数里

```buildoutcfg
class DTree(object):
    def __init__(self, epsilon=0.0001):
        self.tree = Node()
        self.epsilon = epsilon
       
    def fit(self, X_train, Y_train):
        A_recorder = np.arange(X_train.shape[1])
        self._train(X_train, Y_train, self.tree, A_recorder)

    def _train(self, A, D, node, AR):
        # 特殊结束条件：若 D 中所有实例属于同一类，决策树成单节点树，直接返回
        if np.any(np.bincount(D) == len(D)):
            node.y = D[0]
            return
        # 特殊结束条件：若 A 为空，则返回单结点树 T，标记类别为样本默认输出最多的类别
        if A.size == 0:
            node.y = np.argmax(np.bincount(D))
            return
        # 1. 计算特征集 A 中各特征对 D 的信息增益，选择信息增益最大的特征 A_g
        max_info_gain, g = self._feature_choose_standard(A, D)
        # 2. 结束条件：如果 A_g 的信息增益小于阈值 epsilon，决策树成单节点树，直接返回
        if max_info_gain <= self.epsilon:
            node.y = np.argmax(np.bincount(D))
            return
        # 3. 对于 A_g 的每一可能值 a_i，依据 A_g = a_i 将 D 分割为若干非空子集 D_i，将当前结点的标签设为 D_i 中出现最多的类别标签
            # 的；然后对第 i 个子节点，以 D_i 为训练集，以 A - {A_g} 为特征集，递归调用以上步骤，得到子树 T_i，返回 T_i
        node.label = AR[g]
        a_cls = np.bincount(A[:, g])
        new_A, AR = np.hstack((A[:, 0:g], A[:, g+1:])), np.hstack((AR[0:g], AR[g+1:]))
        for k in range(len(a_cls)):
            a_row_idxs = np.argwhere(A[:, g] == k).T[0].T
            child = Node(k)
            node.append(child)
            A_child, D_child= new_A[a_row_idxs, :], D[a_row_idxs]
            self._train(A_child, D_child, child, AR)
    
    def _cal_prob(self, D):
        statistic = np.bincount(D)
        prob = statistic / np.sum(statistic)
        return prob
```


，其中 _cal_prob 函数是为了后文各个不同算法用来实现特征标准函数 _feature_choose_standard 的工具函数。

##### ID3算法

ID3算法应用信息增益准则选择特征。在银行贷款的例子中，希望能快速有一个明确的决定，贷款还是不贷，这样好给客户一个明确的答复。所以更通用地将，我们的目的是：

**先找到最具决定性作用的特征作为判断标准，让决策不确定性尽可能大的减少。**

那么对于上面的重点标出的这句话，我们引出3个重要的概念，然后换用一个更加专业的表述重新表述上面这句话，

**熵**是衡量决策不确定性的指标，**条件熵**衡量了给定某个特征条件下的决策不确定性，**信息增益**衡量了决策不确定性的减少程度。所以换句话说，我们的目的是**找到一个让信息增益最大增加的特征，而信息增益就是熵与条件熵的差**，即，信息增益 g(D,A)、熵 H(D)、条件熵 H(D|A) 满足以下公式，

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=g(D,&space;A)=H(D)-H(D&space;|&space;A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(D,&space;A)=H(D)-H(D&space;|&space;A)" title="g(D, A)=H(D)-H(D | A)" /></a>
</center>

信息增益大表明信息增多，信息增多，则不确定性就越小，就越有利于达到分类的目的。

```buildoutcfg
class DTreeID3(DTree):

    def _feature_choose_standard(self, A, D):
        row, col = A.shape
        prob = self._cal_prob(D)
        prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
        entropy = -np.sum(prob * np.log2(prob))
        max_info_gain_ratio = None
        g = None
        for j in range(col):
            a_cls = np.bincount(A[:, j])
            condition_entropy = 0
            for k in range(len(a_cls)):
                a_row_idxs = np.argwhere(A[:, j] == k)
                # H(D)
                prob = self._cal_prob(D[a_row_idxs].T[0])
                prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
                H_D = -np.sum(prob * np.log2(prob))
                # H(D|A)=SUM(p_i * H(D|A=a_i))
                condition_entropy += a_cls[k] / np.sum(a_cls) * H_D
            feature_choose_std = entropy - condition_entropy
            if max_info_gain_ratio is None or max_info_gain_ratio < feature_choose_std:
                max_info_gain_ratio = feature_choose_std
                g = j
        return max_info_gain_ratio, g
```                                                                                       

##### C4.5算法

C4.5 与 ID3 的第一个差别在于信息增益的标准不一样，
将 `feature_choose_std = entropy - condition_entropy` 这一句替换成 `condition_entropy += a_cls[k] / np.sum(a_cls) * H_D` 即可。

##### CART算法

与 ID3 和 C4.5 算法不一样的是，CART 采用的是不停地二分特征离散值集合的方法。在树形态上的差别在于，前者是一个每层节点数目不定的k叉树，后者是二叉树。

举个例子，比如对于特征 A，有 3 个离散可选值，{A1, A2, A3}，
在 ID3、C4.5，特征A被选取建立决策树节点，如果它有3个类别A1,A2,A3，我们会在决策树上建立一个三叉点，这样决策树是多叉树，而在 ID3、C4.5 中，特征 A 只会只会参与一次划分。。
而 CART 会考虑把特征 A分成 {A1}和{A2,A3}、 {A2}和{A1,A3}、 {A3}和{A1,A2}三种情况，找到基尼系数最小的组合，比如{A2}和{A1,A3}，然后建立二叉树节点，
一个节点是 A2 对应的样本，另一个节点是 {A1,A3} 对应的样本。这意味着后续还有机会继续将 {A1,A3} 二分为 {A1}和{A3}。

```buildoutcfg
class DTreeCART(DTreeID3):

    def _train(self, A, D, node, AR):
        self.visited_set = set()
        self._train_helper(A, D, node, AR)

    def _train_helper(self, A, D, node, AR):
        # 1. 结束条件：若 D 中所有实例属于同一类，决策树成单节点树，直接返回
        if np.any(np.bincount(D) == len(D)):
            node.y = D[0]
            return
        # 2. 与 ID3, C4.5 不一样, 不会直接去掉 A
        if A.size == 0:
            node.y = np.argmax(np.bincount(D))
            return
        # 3. 与 ID3, C4.5 不一样, 不仅要确定最优切分特征，还要确定最优切分值
        max_info_gain, g, v, a_idx, other_idx = self._feature_choose_standard(A, D)
        if (g, v) in self.visited_set:
            node.y = np.argmax(np.bincount(D))
            return
        self.visited_set.add((g, v))
        # 4. 结束条件：如果 A_g 的信息增益小于阈值 epsilon，决策树成单节点树，直接返回
        if max_info_gain <= self.epsilon:
            node.y = np.argmax(np.bincount(D))
            return
        # 5. 与 ID3, C4.5 不一样, 不是 len(a_cls) 叉树，而是二叉树
        node.label = AR[g]
        idx_list = a_idx, other_idx
        for k, row_idx in enumerate(idx_list):
            row_idx = row_idx.T[0].T
            child = Node(k)
            node.append(child)
            A_child, D_child = A[row_idx, :], D[row_idx]
            self._train_helper(A_child, D_child, child, AR)

    def _feature_choose_standard(self, A, D):
        row, col = A.shape
        min_gini, g, v, a_idx, other_idx = None, None, None, None, None
        for j in range(col):
            a_cls = np.bincount(A[:, j])
            # 与 ID3, C4.5 不一样,不仅要确定最优切分特征，还要确定最优切分值
            for k in range(len(a_cls)):
                # 根据切分值划为两类
                a_row_idxs, other_row_idxs = np.argwhere(A[:, j] == k), np.argwhere(A[:, j] != k)
                # H(D) = -SUM(p_i * log(p_i))
                a_prob, other_prob = self._cal_prob(D[a_row_idxs].T[0]), self._cal_prob(D[other_row_idxs].T[0])
                a_gini_D, other_gini = 1 - np.sum(a_prob * a_prob), 1 - np.sum(other_prob * other_prob)
                # H(D|A)=SUM(p_i * H(D|A=a_i))
                gini_DA = a_cls[k] / np.sum(a_cls) * a_gini_D + (1 - a_cls[k] / np.sum(a_cls)) * other_gini
                if min_gini is None or min_gini > gini_DA:
                    min_gini, g, v, a_idx, other_idx = gini_DA, j, k, a_row_idxs, other_row_idxs

        return min_gini, g, v, a_idx, other_idx
```

CART 算法的特点就在于，与 ID3, C4.5 不一样, 不仅要确定最优切分特征，还要确定最优切分值；而确定这个最有切分值后，意味着树的结构是二叉树。

### 三、表现效果

还是贷款的例子，原任务是由 15 个样本组成的训练集，本文多加一个噪声样本（即错误的样本），看是否对模型的训练起到了干扰作用。将上述数据作为训练集建立三种不同的决策树模型，在训练集上的表现效果如下所示，首先是 ID3 算法，

```buildoutcfg
====================DTreeID3====================

<Tree Strucutre>
None+2 
    None+1 
        None+0 
            None+3 
                0+None 
                0+None 
            0+None 
            0+None 
        1+None 
    1+None 

<Label Groundtruth>
[0 0 1 1 0 0 0 1 1 1 1 1 1 1 0 1]

<Label Output>
[0 0 1 1 0 0 0 1 1 1 1 1 1 1 0 0]
```

其中结构中比如第一行`None+2`分别表示`分类标签+决策特征index`，可以看到首先根据第2维特征可以划分出标签为1的样本，在数据集情景下表示有房就必然可以贷款，
；然后根据第1维特征也可以划分出标签为1的样本，意味着有工作则必然可以贷款；然后与维度0和维度3的特征无关，表示是否可以贷款与年龄以及信贷情况无关。
从这里可以看到，决策树方法是具备良好特征选择作用的，模型也十分容易解释。当然该模型只是针对本文上述少量样本构成的数据集，与实际情况无关。

接下来是 C4.5 算法，

```buildoutcfg
====================DTreeC45====================

<Tree Strucutre>
None+2 
    None+1 
        None+0 
            None+3 
                0+None 
                0+None 
            0+None 
            0+None 
        1+None 
    1+None 

<Label Groundtruth>
[0 0 1 1 0 0 0 1 1 1 1 1 1 1 0 1]

<Label Output>
[0 0 1 1 0 0 0 1 1 1 1 1 1 1 0 0]
```

与 ID3 算法的结论完全一致。

最后是 CART算法，

```buildoutcfg
====================DTreeCART====================

<Tree Strucutre>
None+2 
    None+1 
        None+0 
            None+3 
                0+None 
                0+None 
            0+None 
        1+None 
    1+None 

<Label Groundtruth>
[0 0 1 1 0 0 0 1 1 1 1 1 1 1 0 1]

<Label Output>
[0 0 1 1 0 0 0 1 1 1 1 1 1 1 0 0]
```

可以看到虽然三种方法的预测效果是一致的，但 CART 算法的树结构 与 ID3 和 C4.5 算法的树结构有一点区别，CART 得到树是纯二叉树，而ID3 和 C4.5 算法不是。 

1. [wiki: 决策树学习](https://zh.wikipedia.org/wiki/%E5%86%B3%E7%AD%96%E6%A0%91%E5%AD%A6%E4%B9%A0)
2. [csdn: 决策树（ID3、C4.5、CART、随机森林）](https://blog.csdn.net/gumpeng/article/details/51397737)
3. [cnblogs: 决策树（Decision Tree）-决策树原理](https://www.cnblogs.com/huangyc/p/9734972.html)