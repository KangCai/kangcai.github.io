---
layout: post
title: "游戏AI参考 II"
subtitle: "DOTA2 5v5 AI - OpenAI Five"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - 游戏AI
---

> 为了研究高水平游戏AI的开发，利用业余时间里观摩一下目前国际上最高水平游戏AI是怎么做的。本文的内容是2018年上半年颇受关注的 OpenAI 团队做的 DOTA2 5v5 AI，名为 OpenAI-Five。OpenAI 团队在2017年年中做的 DOTA2 影魔 solo AI 能够完胜职业选手，而2018年的 AI 也达到了平均6500天梯分（最近情况不清楚了，大概世界前1000名吧），总之水平很高。本文期望通过对它进行研究，尝试发掘一些可供 AI 开发参考的有效信息。

> 文章首发于[我的博客](https://kangcai.github.io/)，转载请保留链接 ;)

[传送门：OpenAI-Five Blog](https://blog.openai.com/openai-five/)

### 收获

其实也是一些重要的细节信息，

1. OpenAI-Five 从随机权重开始训练，为了避免 “策略崩溃”，**80%的游戏与自己战斗，20%的游戏与过去的自己战斗，来进行强化学习迭代**；
2. Delay Reward 的问题是通过**最大化未来 reward 的指数 γ 衰减总和**来做的；
3. AI 的团队性做单独的团队策略机制，而是通过**团队平均 reward 的形式来影响个体 AI 的决策**；
4. **通过超参数来决定 AI 是更在乎个体 reward 还是团队平均 reward**，博客原文里有这么一句 “We anneal its value from 0 to 1 over training” ，看起来在训练的时候都尝试过，选一个表现最好的超参数。有一个疑问，这个超参数看起来应该在一场比赛中发生变化才合理吧，比如前期更重视个体 reward，后期更重视团队 reward，不知道这句原文是不是这个意思；
5. 原文说 DOTA2 游戏环境一次 tick 花费几毫秒，而 OpenAI Five又是**每4帧获取一个样本，意味着实际运行时每10毫秒~100毫秒获取一次样本**，具体不清楚；
6. 原文说每天相当于打了180年，故可以看出同一时间大约**并行跑了6480场比赛**；
7. 使用**PPO算法**进行强化学习，其中**Actor网络结构为：输入网络 + 共享网络 + 输出网络。其中，输入网络是层次式的 各种State（约20000维度） + FC-relu + concat + max-pool 的结构，共享网络是经过1024个节点的 LSTM，最后的输出分为 操作、技能偏移X、技能偏移Y、移动目标点X、移动目标点Y、传送目标、操作延迟帧数、选择目标 共8大项，每一项都是一个 FC + Softmax 的分类（不连续）输出网络**；
8. 输入特征中，**位置信息用的是绝对位置**，还用到了**单位类型，动作，当然两个都是Embedding**的，**单位状态**当然也必须有，但有三个特征值得特别注意下：**正在攻击的敌方和正被敌方或友方英雄攻击的信息（被友方英雄攻击应该是反补）、最近12帧的血量信息（应该是指短时间内的掉血情况吧）、距离所有友方和敌方的距离#**；
9. 输入网络中，**不定数量的单位状态的输入处理，是通过 FC 之后的 max-pool 来合并的，表明所有单位使用的是同一套网络和参数**，而 max-pool 每次都针对性的 BP 迭代 max-pool 中 max 位置对应输出的上一层单位网络；
10. 输出网络中，**操作输出网络在 FC 层输出向量点乘了当前可使用操作的热编码向量，选敌输出网络在 FC 输出向量点乘了单位的attantion keys**；
11. OpenAI的**APM是 150-170**；
12. **二元奖励效果更好**。不仅仅包含最后的胜利，如果还包含了中间的小奖励，训练更加平稳，效果更好；
13. **没有通过分层强化学习来做，而是直接有5分钟的半衰期的 reward 衰减系数来实现长时间预期**；
14. **技能加点和物品购买是脚本写的**；
15. 根据公布的信息推测1天大约跑23万场比赛，共6天，故**总训练场次约140万场**。
16. 具体 reward 信息是 DOTA2 专家给出的，微调了几版，如下所示，

**16.1 属性reward**

| 属性 | 权重 | 描述                |
| ---------- | ------ | -------------------------- |
| Experience | 0.002  | 一经验值 |
| Gold       | 0.006  | 一个金币 |
| Mana       | 0.75   | 蓝量变化百分比 |
| Hero Health| 2.0    | 血量变化百分比（从0到1的二次函数） |
| Last Hit   | 0.16   | 一个正补 |
| Deny       | 0.2    | 一个反补 |
| Kill       | -0.6   | 击杀一个英雄 |
| Death      | -1.0   | Dying. |

击杀一个英雄的 reward 之所以是负值，是因为

**16.2 建筑reward**

是随着血量百分比的线性函数: 

    建筑reward = 权重 * (1 + 2 * 血量百分比).

| 建筑  | 权重 |
| ---------- | ------ |
| 圣坛     | 0.75   |
| 一塔 | 0.75   |
| 二塔 | 1.0    |
| 三塔 | 1.5    |
| 门牙塔 | 0.75   |
| 兵营   | 2.0    |
| 基地 | 2.5    |

The agent receives extra reward upon killing several special buildings near the end of the game:

| Extra Team | Weight | 描述                 |
| ---------- | ------ | ----------------------------- |
| 超级兵      | 4.0    | 打掉最后一个兵营 |
| 胜利       | 2.5 | 获胜 |

**16.3 分路reward**

游戏开始时给各个英雄预先分配一条 “线”（分路），如果离开这条路就会收到 0.02 的惩罚，以此来对 AI 训练出 “线” 的概念。
In addition to the above reward signals, our agent receives a special reward to encourage exploration called "lane assignments." During training, we assign each hero a subset of the three lanes in the game. The model observes this assignment, and receives a negative reward (-0.02) if it leaves the designated lanes early in the game. This forces the model to see a variety of different lane assignments. During evaluation, we set all heroes' lane assignments to allow them to be in all lanes.

**16.4 reward零和**

每个英雄的 reward 都要减去敌方队伍 reward 的均值:

    hero_rewards[i] -= mean(enemy_rewards)

主要是防止两边找到双赢的方式玩游戏，实际上是 reward 设置的不完美，但这个问题暂时无解，所以通过这种方式来增加训练的容错率。

**16.5 奖励随时间的缩放**

为了突出前期的重要性，故通过下面的公式来扩大前期 reward，减少后期 reward:

    hero_rewards[i] *= 0.6 ** (T/10 min)

### 总结

总的来说，OpenAI-Five的方法还是比较朴素的。一般来说，对于复杂的强化学习应用场景，通常有如下问题以及相应的解决方案：

1. 状态空间大：解决方法如先用World Models抽象，再进行决策。
2. 局面不完全可见：一般认为需要进行一定的搜索，如AlphaGo的MCTS（蒙特卡洛树搜索）。
3. 动作空间大：可以使用模仿学习(Imitation Learning)，或者与层次强化学习结合的方法。
4. 时间尺度大：一般认为需要时间维度上的层次强化学习(Hierarchical Reinforcement Leanring)来解决这个问题。

神奇的是，OpenAI 没有使用上述任一方法，而仅仅使用高 γ 值的PPO基础算法，就解决了这些问题，这说明凭借非常大量的计算，强化学习的基础算法也能突破这些挑战。换个角度看，WorldModels、MCTS、IL、HRL等方法是学术界研究的重点方向，而 OpenAI-Five 却没有使用，也是 OpenAI-Five 潜在的提升空间。这些更高效的方法若被合理应用，可以加快模型的学习速度，增强模型的迁移能力，并帮助模型突破当前的限制。

[传送门：OpenAI Five Actor 网络模型](https://kangcai.github.io/img/in-post/post-ml/OpenAI_Five_Model.jpg)

**参考文献**
[OpenAI-Five Dota2 reward](https://gist.github.com/dfarhi/66ec9d760ae0c49a5c492c9fae93984a)
[技术架构分析：攻克Dota2的OpenAI-Five](http://www.qianjia.com/html/2018-06/28_296463.html)