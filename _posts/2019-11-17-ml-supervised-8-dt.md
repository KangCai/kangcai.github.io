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

跟前一篇介绍最大熵模型用的示例一样，本文仍采用银行贷款资质判定的这个例子来对决策树进行解释，如下表所示，前 4 列属性，包括 “年龄”、是否“有工作”、是否“有房”、“信贷情况”是否良好，是 4 个维度的特征，银行根据该 4 个特征来判定是否批准贷款。表格中一共有15个样本，表格最后一列是“是否批准贷款”的实际结果，作为训练标签，

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

针对这个问题，我们用一个十分简单的 if else 语句来解决，可以画出以下的决策树。

但很明显，这个解并不是最优模型，那有什么办法能找到一个最优模型呢？即便是根据我们的直觉找到了所谓的最优模型，那么对于所有问题是否有一种通用找最优模型的方法呢？下面将针对该问题来引出 3 种经典决策树方法: ID3算法、C45算法、CART算法。

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
        # 3. 对于 A_g 的每一可能值 a_i，依据 A_g = a_i 将 D 分割为若干非空子集 D_i，将当前结点的标记设为样本数最大的 D_i 对应
            # 的类别，即对第 i 个子节点，以 D_i 为训练集，以 A - {A_g} 为特征集，递归调用以上步骤，得到子树 T_i，返回 T_i
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

**找到最具决定性作用的特征，作为判断标准，让决策不确定性尽可能大的减少。**

那么对于上面的重点标出的这句话，我们引出3个重要的概念，然后换用一个更加专业的表述重新表述上面这句话，

决策不确定性的衡量的指标就是**熵**，给定某个特征条件下的决策不确定性就是**条件熵**，决策不确定性的减少对应的就是**信息增益**。所以换句话说，我们的目的是**找到一个让信息增益最大增加的特征，其中信息增益就是熵与条件熵的差**，即，信息增益 g(D,A)、熵 H(D)、条件熵 H(D|A) 满足以下公式，

<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=g(D,&space;A)=H(D)-H(D&space;|&space;A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(D,&space;A)=H(D)-H(D&space;|&space;A)" title="g(D, A)=H(D)-H(D | A)" /></a>
</center>

信息增益大表明信息增多，信息增多，则不确定性就越小，就越有利于分类目的。

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

##### C45算法

总体来说，

```buildoutcfg
class DTreeC45(DTree):

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
                # H(D) = -SUM(p_i * log(p_i))
                prob = self._cal_prob(D[a_row_idxs].T[0])
                prob = np.array([a if 0 < a <= 1 else 1 for a in prob])
                H_D = -np.sum(prob * np.log2(prob))
                # H(D|A)=SUM(p_i * H(D|A=a_i))
                condition_entropy += a_cls[k] / np.sum(a_cls) * H_D
            feature_choose_std = entropy / (condition_entropy + 0.0001)
            if max_info_gain_ratio is None or max_info_gain_ratio < feature_choose_std:
                max_info_gain_ratio = feature_choose_std
                g = j
        return max_info_gain_ratio, g
```

##### CART算法

```buildoutcfg
class DTreeCART(DTree):

    def _feature_choose_standard(self, A, D):
        row, col = A.shape
        min_gini = None
        g = None
        for j in range(col):
            a_cls = np.bincount(A[:, j])
            gini_DA = 0
            for k in range(len(a_cls)):
                a_row_idxs = np.argwhere(A[:, j] == k)
                # H(D) = -SUM(p_i * log(p_i))
                prob = self._cal_prob(D[a_row_idxs].T[0])
                gini_D = 1 - np.sum(prob * prob)
                # H(D|A)=SUM(p_i * H(D|A=a_i))
                gini_DA += a_cls[k] / np.sum(a_cls) * gini_D
            if min_gini is None or min_gini > gini_DA:
                min_gini = gini_DA
                g = j
        return min_gini, g
```

##### CART回归树

```buildoutcfg
class DTreeRegressionCART(object):

    def __init__(self, max_depth=1):
        self.tree = Node()
        self.max_depth = max_depth

    def fit(self, X_train, Y_train):
        A_recorder = np.arange(X_train.shape[1])
        self._train(X_train, Y_train, self.tree, A_recorder)

    def predict(self, X):
        n = X.shape[0]
        Y = np.zeros(n)
        for i in range(n):
            Y[i] = self.tree.predict_regression(X[i, :])
        return Y

    def _train(self, A, D, node, AR, depth=0):
        # 1. 结束条件：到最后一层 | A 或 D 一样
        if depth == self.max_depth or np.all(D == D[0]) or np.all(A == A[0]):
            node.y = np.mean(D)
            return
        # 2. 选择第j个变量A_j（切分变量splitting variable）和 切分点s（splitting point）
        min_f, min_j, min_s, min_idx1, min_idx2 = None, None, None, None, None
        row, col = A.shape
        for j in range(col):
            a_col = A[:, j]
            # 这里实现比较简化，s 就直接取最值的平均数
            s = (np.max(a_col) + np.min(a_col)) * 0.5
            R1_idx, R2_idx = np.argwhere(a_col <= s).T[0], np.argwhere(a_col > s).T[0]
            if R1_idx.size == 0 or R2_idx.size == 0:
                continue
            c1, c2 = np.mean(D[R1_idx]), np.mean(D[R2_idx])
            f1, f2 = np.sum(np.square(D[R1_idx] - c1)), np.sum(np.square(D[R2_idx] - c2))
            if min_f is None or min_f > f1 + f2:
                min_f, min_j, min_s, min_idx1, min_idx2 = f1 + f2, j, s, R1_idx, R2_idx
        if min_f is None:
            node.y = np.mean(D)
            return
        # 3. 向下一层展开
        node.label, node.s = AR[min_j], min_s
        for i, idx_list in enumerate((min_idx1, min_idx2)):
            child = Node(i)
            node.append(child)
            self._train(A[idx_list, :], D[idx_list], child, AR, depth+1)
```
### 三、表现效果

还是贷款的例子



下面来具体介绍 熵、条件熵 以及 信息增益 这3个概念。




1. [wiki: 决策树学习](https://zh.wikipedia.org/wiki/%E5%86%B3%E7%AD%96%E6%A0%91%E5%AD%A6%E4%B9%A0)
2. [csdn: 决策树（ID3、C4.5、CART、随机森林）](https://blog.csdn.net/gumpeng/article/details/51397737)
3. [cnblogs: 决策树（Decision Tree）-决策树原理](https://www.cnblogs.com/huangyc/p/9734972.html)