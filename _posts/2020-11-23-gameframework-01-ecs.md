---
layout: post
title: "【游戏框架】1-ECS结构"
author: "Kang Cai"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.4
tags:
  - GamePlay
---

> 算法。文章首发于[我的博客](https://kangcai.github.io)，转载请保留链接 ;)

一、ECS 框架

ECS（Entity-Component-System）

1.1 面向接口编程

ECS就是典型的面向接口编程（Data Access Object，DAO模式)，思想是属性决定类型。而面向对象编程（Object Oriented Programming，OOP），思想是类型包含属性。举个例子，如果用面向对象的思想来定义手机，那么有，

手机 = 天线 + 话筒 + 麦克风

有了天线、话筒、麦克风，我们可以打电话，那么这个打电话的设备是手机，进一步发展，我们为手机增加了更多的属性，就有了智能手机，

智能手机 = 屏幕 + 可视化交互系统 + 天线 + 话筒 + 麦克风

现在科技发展迅速，手环、手表、电脑 这些都有天线、话筒、麦克风，都有打电话功能，以后可能 衣服、帽子 都能打电话，用面向对象的思想定义就有了局限性。解决这个问题的一种有效方案就是用面向接口编程。ECS 很能体现出 has-a 优于 is-a，不用为了后发的共性调整对象树。

1.2《守望先锋》的例子

守望先锋实现的 ECS 框架有以下几个规则：system内无状态，component内无函数，system间禁止互相调用，system间有共享Utility函数使用，system中的遍历通过 “主component+它的兄弟姐妹component” 的方式实现，允许不少component是单例component。

服务端 `Components：AIBehavoir, AIBot, AIInfluence, AIMovement, AIPathfinder, AIPointFinder, AITarget, AITargeting, AITuning, Area, Assist, Breakable, CaptureArea, CharacterController, CharacterMover, Connection, DamageArea, DerpBot, DynamicObstacle, Escort, FilterBits, GameDebug, GameMode, GameModeParticipant, GmpStats, Health, HeroProgression, HotReload, IdlePathdataInvalidate, InputStream, MirroredIdleAnim, Model, ModifyHealthQueue, MovementState, MovingPathdataInvalidate, MOVEING_PALTFORM, NetRelevancy, NetworkQuality, ParametricMover, PathdataInvalidate, Preload, Pet, PetMaster, PlayerOfTherGame, PlayerAchievements, PlayerProgression, Possessable, Possessor, Prediction, Projectile, ProjectileModifier, PVPGameLegacy, Replication, RigidBodyMover, ...`

服务端 `Systems：NetworkEvent, InstanceClientDebugMessage, InstanceClientMessage, Observer, Spectator, PathdataInvalidate, FixedUpdate, DerpBot, Debug, MovementVolume, AIStrategic, AIPathFind, AIBehavior, AISpawn, AIMovement, UnsynchronizedMovement, MovementState, MovementExertion, AIPerception, Statescript, Weapon, Anim, PvPLegacy, Combat, World, GameMode, AIPointFind, StatescriptPostSim, Predictor, Stats, Hero, SeenBy, Voice, Mirror, Possession, NetworkMessage, ServerBreakable, NetworkFlow, InterpolateMovementState, SpatialQuery, Replay, GameModerator`

客户端 Systems：`LiveGameInput，NetworkEvent, Network, WorkState, GameMode, PvPLegacy, Mirror, PathdataInvalidate, WeaponStaging, ViewTarget, Possession, Command, MovementState, SimpleMovement, RigidBodyMovement, UnsynchronizedMovement, LocalPlayerMovement, MovementExertion, WeaponAim, Statescript, Weapon, Debug, ResponsivenessDebug, StatescriptPostSim, Anim, StatescriptUX, FinishAsyncWorkAnim, WeaponPostSim, Combat, IdleAnim, MoveEffect, SpatialQuery, Camera, FirstPerson, Map, Sound, Voice, LocalHitEffect, HeroFullBodyEffects, UpdateSceneViewFlags, ResolveContact, ClientInstanceDebugMessage, World, NetworkMessage, NetworkFlow, UXGame`

客户端 Components：与服务端 Components 比较类似，略。

遵守的规则：

【!】Component 不能有改变属性的函数；

【!】System 不能有状态；

【!】只有少量的 System 需要改变 Component 状态，这种情况下它们必须自己管理复杂性；

【!】单例 Component 通过在 EntityAdmin 中直接访问；

【!】多个 System 调用的同一个函数，写在 Util 文件里，作为公共函数使用，但是要注意，必须控制 Util 函数依赖很少的 Component，而且不应该带副作用或者很少的副作用（改变Component 属性），而且如果 Util 函数依赖很多 Component，就需要限制调用点；

如果只有一个调用点，那么行为的复杂性就会很低，因为所有的副作用都限定到函数调用发生的地方了。

好的单例组件可以通过“推迟”（Deferment）来解决System间耦合的问题。“推迟”存储了行为所需状态，然后把副作用延后到当前帧里更好的时机再执行。比如，你用猎空的枪去射击一面墙，留下了一堆麻点，然后法老之鹰发出一枚火箭弹，在麻点上面造成了一个大面积焦痕。你肯定想删了那些麻点，要不然看起来会很丑，像是那种深度冲突（Z-Fighting）引起的闪烁。

代码示例：

System update 的方式如下，

```buildoutcfg
void EntityAdmin::Update(f32 temeStep) { 
    for (System* s : m_system) { 
        s->Update(timeStep) 
    } 
} 

... 

void SomeSystem::Update(f32 timeStep) override { 
    for (DerpComponent* d: ComponentItr<DerpComponent>(m_admin)) { 
        d->m_timeAccu += timeStep; 
        if (d->m_timeAccu > d->m_timeToHerp) { 
            this->HerpYourDerp(d, d->Sibling<HerpComponent>()); 
        } 
        SomeUtil::XXX(d->A, XXX); 
    } 
} 
```
