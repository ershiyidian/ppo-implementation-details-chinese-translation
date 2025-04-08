# ICLR Blog Track | 博客文章
# PPO（Proximal Policy Optimization，近端策略优化）的 37 项实现细节

**2022 年 3 月 25 日 | 标签：proximal-policy-optimization, reproducibility, reinforcement-learning, implementation-details, tutorial**  
作者：Huang, Shengyi；Dossa, Rousslan Fernand Julien；Raffin, Antonin；Kanervisto, Anssi；Wang, Weixun

---

## 引子

Jon 是一名研究生一年级的学生，对强化学习（RL）非常感兴趣。在他看来，强化学习令人着迷，因为他可以使用像 Stable-Baselines3（SB3）这样的 RL 库来训练智能体去玩各种游戏。他很快发现了 Proximal Policy Optimization (PPO) 这类算法速度快、适用性广，于是想自己动手实现 PPO 以作为一个学习过程。读完论文后，Jon 想道：“嗯，看上去挺直观嘛。” 然后他打开了一个代码编辑器，开始编写自己的 PPO。对于环境，他先选择了 Gym 里的 CartPole-v1，不久之后，Jon 让 PPO 在 CartPole-v1 上成功运行。他玩的很开心，并且想进一步让他的 PPO 能在更有趣的环境上工作，比如 Atari 游戏和 MuJoCo 机器人任务。他心想：“那可太酷了吧！”

然而，他很快遇到了困难。让 PPO 在 Atari 和 MuJoCo 上成功运行比他想象中要复杂得多。Jon 开始在网上寻找参考实现，但随后他就被各种非官方代码仓库给淹没了：它们实现细节各不相同，而官方仓库又是 TensorFlow 1.x 代码，Jon 看起来很费力。所幸，Jon 偶然看到了两篇关于 PPO 实现细节的论文，最近才发表。“就是它了！”他笑逐颜开。尽管兴奋不已，他却还是不禁有很多疑惑。于是，Jon 在办公室里四处奔跑，结果不小心撞到了正在从事强化学习相关工作的 Sam。他们展开了下面的对话：

> **Jon**： “嘿，我刚看了那篇 `implementation details matter` 论文，以及那篇 `what matters in on-policy RL` 论文。真是干货十足啊。我早就觉得 PPO 没那么简单！”
>
> **Sam**： “没错，PPO 确实不好整，我也很喜欢这两篇聚焦于各种实现细节的论文。”
>
> **Jon**： “确实。我感觉现在对 PPO 理解更透彻了。你最近一直在用 PPO，对吗？问我几个问题考考我吧！”
>
> **Sam**： “好啊。如果你在 Breakout 这款 Atari 游戏里运行官方 PPO，智能体大约在 4 小时左右就能拿到 400 分。你知道官方的 PPO 是怎么做到的吗？”
>
> **Jon**： “嗯……这个问题还挺好。我记得那两篇论文里好像确实没怎么提到这些。”
>
> **Sam**： “另一个问题，`procgen` 论文里做过使用官方 PPO+LSTM 的实验。你知道 PPO + LSTM 是怎么实现的吗？”
>
> **Jon**： “呃……我对 PPO + LSTM 了解不多。”
>
> **Sam**： “官方的 PPO 还能支持 MultiDiscrete（多离散）动作空间，允许用多个离散值来描述一个动作。你知道这是怎么实现的吗？”
>
> **Jon**： “…（沉默）”
>
> **Sam**： “最后，如果你手头只有一些常见的开发工具（比如 numpy、gym…）和某个神经网络库（torch、jax…），你能完全从头写出 PPO 吗？”
>
> **Jon**： “呃……听起来很难啊。之前那几篇论文虽然分析了实现细节，但是并没有展示如何把这些点整合到代码中。另外，我也意识到它们所做的结论大多是基于 MuJoCo 任务，而并不一定能迁移到像 Atari 这种环境。唉，我现在有点儿沮丧……”
>
> **Sam**： “别灰心，PPO 本来就复杂。对了，我最近在做一些从零开始实现 PPO 的视频教程，还有篇博客会更详细地解释。”

就是这篇博客！不同于做消融研究、比较哪些细节最重要，这篇博客退一步，着重于 **在所有方面** 再现 PPO 的结果。更具体地说，本博客将以下面几种方式来补充之前的工作：

1. **代码谱系分析**：我们通过研究官方 PPO 实现（`openai/baselines` GitHub 仓库）的历史版本，来明确“复现官方 PPO”的含义。我们将看到 `openai/baselines` 里的代码经历过几次重构，因此它可能与原论文产生不同的结果。弄清楚哪个版本的官方实现值得深入研究是很重要的。
2. **视频教程和单文件实现**：我们制作了如何在 PyTorch 中从头实现 PPO 的逐行讲解视频，并且在代码仓库中提供了单文件的代码示例，以尽可能提高可读性。这些实现包含了官方 PPO 中应对经典控制任务、Atari 游戏和 MuJoCo 任务的所有细节。视频如下所示：
   - （此处原文插入了视频链接/预览）
3. **含参考链接的实现清单**：在我们重实现 PPO 的过程中，我们罗列了 37 项实现细节，并给出了对应的永久性链接到具体代码位置（在学术论文中很少做到），同时也指明了对应的文献出处。这 37 项细节包括：
   - 13 个核心实现细节
   - 9 个 Atari 专用实现细节
   - 9 个机器人学任务（连续动作空间）相关实现细节
   - 5 个 LSTM 相关实现细节
   - 1 个 MultiDiscrete 动作空间的实现细节
4. **高保真再现**：为了验证我们的复现程度，我们在经典控制任务、Atari 游戏、MuJoCo 任务、LSTM 以及实时策略（RTS）游戏任务上做了实验，结果与官方实现非常接近。
5. **特定场景下才会用到的实现细节**：我们还额外介绍了官方实现中并未采用、但在某些特殊场景可能有用的 4 条实现细节。

我们的最终目标是帮助大家彻底理解 PPO 的实现细节，并能够高保真地复现已有结果，进而为新的研究提供灵活的定制能力。为了便利可重复研究，我们在 [GitHub 仓库](https://github.com/vwxyzjn/ppo-implementation-details) 开源了全部源代码，并且在 [Weights and Biases 项目](https://wandb.ai/vwxyzjn/ppo-details) 中对全部实验进行了可追踪记录。

---

## 背景

PPO（Proximal Policy Optimization）是由 Schulman 等人在 2017 年提出的一种策略梯度算法。它是对 Trust Region Policy Optimization (TRPO) (Schulman et al., 2015) 的进一步改进，用一个更简单的 **裁剪（clipped）目标函数** 取代了 TRPO 中代价高昂的二阶优化步骤。尽管目标函数更简单，PPO 依然在多种控制任务中展现出了比 TRPO 更高的采样效率。PPO 在 Atari 游戏上也有良好的表现。

为了让研究更透明，Schulman 等人（2017）将 PPO 的源码以 `pposgd` 命名提交到了 `openai/baselines` GitHub 仓库（commit: `da99706`, 时间：2017/7/20）。之后，`openai/baselines` 的维护者又对代码做了一系列修订，主要时间节点如下：

- **2017/11/16**, commit `2dd7d30`：维护者推出了重构后的 `ppo2` 版本，并把原先的 `pposgd` 改名为 `ppo1`。有个 GitHub issue 显示，维护者推荐 `ppo2`，认为它能更好地利用 GPU（因为可以在多个环境上做 batch 处理）。
- **2018/8/10**, commit `ea68f3b`：在做了若干修订后，维护者对 `ppo2` 进行评测并给出了 MuJoCo benchmark。
- **2018/10/4**, commit `7bfbcf1`：在做了若干修订后，维护者对 `ppo2` 进行评测并给出了 Atari benchmark。
- **2020/1/31**, commit `ea25b9e`：维护者合并了最后一次 commit。截至目前（原文发表时），`openai/baselines` 再无新的 commit。`ppo2 (ea25b9e)` 这一版本也成为很多与 PPO 相关的资源的基础：
  - 各大 RL 库，如 Stable-Baselines3 (SB3)、`pytorch-a2c-ppo-acktr-gail`、`CleanRL` 等，都高度遵循了 `ppo2 (ea25b9e)` 的实现细节。
  - 最近的一些论文（Engstrom, Ilyas, et al., 2020；Andrychowicz, et al., 2021）也在 `ppo2 (ea25b9e)` 上研究了机器人学任务的实现细节。

近年来，复现 PPO 结果日渐成为一个挑战。下表展示了多个流行 RL 库在 Atari 和 MuJoCo 环境里所报告的 PPO 最优性能（如原文中的表格所示）：

| RL 库 | GitHub Stars | 基准来源 | Breakout | Pong | BeamRider | Hopper | Walker2d | HalfCheetah |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Baselines pposgd / ppo1 (da99706)** | GitHub stars | paper ($) | 274.8 | 20.7 | 1590 | ~2250 | ~3000 | ~1750 |
| **Baselines ppo2 (7bfbcf1 & ea68f3b)** |  | docs (*) | 114.26 | 13.68 | 1299.25 | 2316.16 | 3424.95 | 1668.58 |
| **Baselines ppo2 (ea25b9e)** |  | 本文 (*) | 409.265 ± 30.98 | 20.59 ± 0.40 | 2627.96 ± 625.751 | 2448.73 ± 596.13 | 3142.24 ± 982.25 | 2148.77 ± 1166.023 |
| **Stable-Baselines3** | GitHub stars | docs (0) (^) | 398.03 ± 33.28 | 20.98 ± 0.10 | 3397.00 ± 1662.36 | 2410.43 ± 10.02 | 3478.79 ± 821.70 | 5819.09 ± 663.53 |
| **CleanRL** | GitHub stars | docs (1) (*) | ~402 | ~20.39 | ~2131 | ~2685 | ~3753 | ~1683 |
| **Tianshou** | GitHub stars | paper, docs (5) (^) | ~400 | ~20 | - | 7337.4 ± 1508.2 | 3127.7 ± 413.0 | 4895.6 ± 704.3 |
| **Ray/RLlib** | GitHub stars | repo (2) (*) | 201 | - | 4480 | - | - | 9664 |
| **SpinningUp** | GitHub stars | docs (3) (^) | - | - | - | ~2500 | ~2500 | ~3000 |
| **ChainerRL** | GitHub stars | paper (4) (*) | - | - | - | 2719 ± 67 | 2994 ± 113 | 2404 ± 185 |
| **Tonic** | GitHub stars | paper (6) (^) | - | - | - | ~2000 | ~4500 | ~5000 |

注解：
- `(-)`：尚无公开指标  
- `($)`：实验采用了 MuJoCo v1 环境  
- `(*)`：实验采用了 MuJoCo v2 环境  
- `(^)`：实验采用了 MuJoCo v3 环境  
- `(0)`：MuJoCo 实验跑 100 万步（1M steps），Atari 实验跑 1000 万步（10M steps），只用 1 个随机种子  
- `(1)`：MuJoCo 实验跑 200 万步（2M steps），Atari 实验跑 1000 万步，随机种子为 2 个  
- `(2)`：Atari 实验跑 2500 万步，10 个 worker（每个 worker 5 个环境）；MuJoCo 实验跑 4400 万步，16 个 worker；随机种子为 1 个  
- `(3)`：300 万步，PyTorch 版本，随机种子 10 个  
- `(4)`：200 万步，随机种子 10 个  
- `(5)`：MuJoCo 实验跑 300 万步，随机种子 10 个；Atari 实验跑 1000 万步，随机种子 1 个  
- `(6)`：500 万步，随机种子 10 个  

从上表可以看出：

1. `openai/baselines` 的几次修订并非没有性能影响。要复现 PPO 的结果之所以难，其中一个原因就是连官方实现本身都可能产出不一致的结果。
2. `ppo2 (ea25b9e)` 以及那些依照其实现细节编写的库，往往报告的结果非常相近。而其他库在性能上可能存在更大的差异。
3. 值得一提的是，很多库报告了在 MuJoCo 上的性能，但没有报告在 Atari 上的。
4. 即便如此，我们还是发现 `ppo2 (ea25b9e)` 值得深入研究。它在 Atari 和 MuJoCo 上都能取得不错成绩，并且支持包括 LSTM、MultiDiscrete 动作空间等高级特性，可以应用于更复杂的场景（如 RTS 游戏）。因此，我们将 `ppo2 (ea25b9e)` 视为“官方 PPO 实现”，并在后文中以此为基础展开。

---

## 如何复现官方 PPO 实现

本节将介绍 5 大类实现细节并提供对应的 PyTorch 从零实现示例：

1. 13 个“核心”实现细节  
2. 9 个 Atari 专用实现细节  
3. 9 个机器人学任务（连续动作空间）相关实现细节  
4. 5 个 LSTM 实现细节  
5. 1 个 MultiDiscrete 动作空间实现细节  

在每个类别（除第一类外），我们都在 3 个环境、3 个随机种子上做了基准测试（与原实现对比），并会给出结果曲线。

---

### 1. 13 个核心实现细节

这是 PPO 中无论在哪种任务都通用的核心实现。为帮助读者更好地用 PyTorch 实现这些细节，我们准备了一个逐行讲解的视频（原文中附链接）。不过，视频里只实现了其中 11 个细节，有 2 个未在视频中出现——因此标题是“11 个核心细节”。

下面列出 13 个细节。

---

#### (1) 向量化环境架构（`common/cmd_util.py#L22`）  
**（代码层面的优化）**

PPO 使用了一种高效的向量化（vectorized）架构，其特征是由单个 Learner 同时收集并从多个环境中学习。伪代码如下所示：

```
envs = VecEnv(num_envs=N)
agent = Agent()
next_obs = envs.reset()
next_done = [0, 0, ..., 0] # 长度为N

for update in range(1, total_timesteps // (N*M)):
    data = []
    # ------- ROLLOUT 阶段 -------
    for step in range(0, M):
        obs = next_obs
        done = next_done
        action, other_stuff = agent.get_action(obs)
        next_obs, reward, next_done, info = envs.step(action)  # 并行地对N个环境执行一步
        data.append([obs, action, reward, done, other_stuff])
    
    # ------- LEARNING 阶段 -------
    agent.learn(data, next_obs, next_done)  # 这里data总长度是N*M
```

- 其中 `envs = VecEnv(num_envs=N)` 表示我们运行了 `N` 个环境，可能是并行多进程，也可能是单进程顺序调用。  
- `envs.reset()` 会一次性返回 `N` 个初始观测，存储在 `next_obs` 里。  
- `next_done` 是一个长度为 `N` 的数组，记录每个子环境是否完成（0 表示未完成，1 表示完成）。  
- 每次我们先执行一个 ROLLOUT 阶段：对 `N` 个环境各走 `M` 步，将所有数据存到 `data`。如果某个子环境在某一步结束（done=true），环境就会自动重置，下一个 `next_obs[i]` 就变成这个子环境新的初始观测，而 `next_done[i]` 被置为 1。  
- 然后在 LEARNING 阶段中，我们基于 `data`（长度 `N*M`）以及 `next_obs, next_done` 来做学习，例如要用 `next_obs` 去做价值的引导，算优势函数（advantages）、回报（returns）等等。  
- 随后进入下一轮更新时，`next_obs` 会成为新一轮 ROLLOUT 的初始观测，`next_done` 也随之继续记录环境是否在新一轮开始处就已结束并重置。

这种设计能让 PPO 不必等待环境真正结束一整条轨迹后再学习，而是可以在每轮采样 `M*N` 步后就更新一次，从而在游戏可能非常长（如 StarCraft II 可达 10 万步单局）的情况下保持可行的记忆需求与计算效率。

实践中常见的错误实现是根据整条回合（episode）来做训练，并设定最大回合长度，如下伪代码：

```
env = Env()
agent = Agent()
for episode in range(1, num_episodes):
    next_obs = env.reset()
    data = []
    for step in range(1, max_episode_horizon):
        obs = next_obs
        action, other_stuff = agent.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        data.append([obs, action, reward, done, other_stuff])
        if done:
            break
    agent.learn(data)
```

这样做有几个缺点：  
1. 每步只能拿到一个环境的样本，效率很低。  
2. 如果游戏 horizon（上限步数）非常大，比如 StarCraft II，单回合就要存储大量数据。  

而向量化环境可以用 `N*M` 步的固定长度分段进行训练，哪怕游戏一局要跑 10 万步，也能分拆为多个小段来学习。

- `N` 即文献中提到的 `num_envs`（在 Andrychowicz et al. (2021) 中是决策 C1），`M*N` 则对应 iteration_size（决策 C2）。  
- 他们发现大 `N` 会提高训练吞吐量，但会使性能下降，因为 `M` 对应的经验片段长度（trajectory segment）就变短了，“更早地去做价值引导”，导致样本效率变低。他们做的是机器人环境。不过，如果我们从实际的“墙上时钟时间效率”来评估，增大 `N` 可能让训练更快结束，你也可以多跑会儿就弥补了损失。比如 Brax 使用了 `N=2048, M=20` 这种巨大的并行度，可以在 1 分钟内解决与 MuJoCo 类似的任务。  
- 此外，向量化环境也可支持多智能体（MARL）：如果一个环境本身就有 `N` 个玩家，就可以将其视为一个 `N=H*K` 的向量化环境，每个玩家一条 obs-action 轨迹，进行自博弈等。  
- 这种多智能体场景在 Gym-μRTS、PettingZoo 等库中应用很多。

---

#### (2) 权重的正交初始化与偏置的常数初始化（`a2c/utils.py#L58`）  
**（神经网络相关，代码层面的优化）**

`openai/baselines` 里有些底层初始化代码是通用的，比如 `a2c/utils.py#L58` 里写了正交（orthogonal）初始化和偏置设为 0 的策略，虽然后来在 PPO 也继承了这部分逻辑。在 Atari 的 CNN 初始化中（`common/models.py#L15-L26`）和 MuJoCo 的 MLP 初始化里（`common/models.py#L75-L103`），都用到了这个。  
- 隐藏层的权重使用正交初始化，缩放系数是 `np.sqrt(2)`，偏置全置 0。  
- 输出层：策略头（policy head）一般用较小的缩放（0.01），价值头（value head）一般用 `1.0`。  
- 注意：`baselines/a2c/utils.py` 里的正交初始化实现和 PyTorch 自带的 `torch.nn.init.orthogonal_` 有一些差异，但影响很小。  

文献中：  
- Engstrom, Ilyas, et al. (2020) 发现正交初始化在提升最高回报方面强于默认的 Xavier 初始化。  
- Andrychowicz, et al. (2021) 发现将策略输出层权重初始化得足够小、让动作分布初始时接近 0（均值为 0），是有助于性能的。

---

#### (3) Adam 优化器的 `epsilon` 参数（`ppo2/model.py#L100`）  
**（代码层面的优化）**

PPO 的默认 `epsilon` 参数是 `1e-5`，而 PyTorch 中 Adam 的默认值是 `1e-8`、TensorFlow 的默认值是 `1e-7`。但在 PPO 代码里却没有显式暴露这个超参数，可见这是一个容易被忽略、但对高保真复现却又不得不匹配的细节。

对比之下：  
- Andrychowicz, et al. (2021) 对 Adam 的参数做了网格搜索，建议把 `beta1=0.9`，并使用 TensorFlow 默认的 `1e-7`。  
- Engstrom, Ilyas, et al. (2020) 用的是 PyTorch 默认的 `1e-8`。

---

#### (4) Adam 学习率退火（annealing）（`ppo2/ppo2.py#L133-L135`）  
**（代码层面的优化）**

PPO 可以选择让 Adam 的学习率恒定或衰减。在 Atari 上，默认超参是从 `2.5e-4` 线性地衰减到 0；在 MuJoCo 上是从 `3e-4` 衰减到 0。

- Engstrom, Ilyas, et al. (2020) 发现学习率退火能提升最终回报。  
- Andrychowicz, et al. (2021) 也发现学习率退火带来微小但正向的收益。

---

#### (5) 一般优势估计（GAE, Generalized Advantage Estimation）（`ppo2/runner.py#L56-L65`）  
**（理论相关）**

虽然 PPO 论文里提到策略梯度的目标中包含优势项，但在实际实现中，它用到了 GAE（Schulman, 2015b）。其中要注意两点：

1. **值函数引导（value bootstrap）**（`ppo2/runner.py#L50`）：如果某个子环境未终止也未截断（truncated），PPO 会对下一个状态的价值进行估计并把它当作回报的一部分。  
2. **对截断（truncation）的处理**：几乎所有 Gym 环境都有时间上限，一旦超时就会返回 `done=True`。但在官方实现里，如果是因为超时导致的截断，它并没有把该终止状态当作终止来处理（并未额外做正确的价值估计）。不过为了追求高保真复现，我们就照官方那样实现。  
3. 最后，PPO 的回报形式是 `returns = advantages + values`，也就是一种 TD(λ) 形式的价值估计（λ=1 时就变成了蒙特卡洛估计）。

- Andrychowicz, et al. (2021) 发现 GAE 在大多数任务上优于 N 步回报。

---

#### (6) 小批次（minibatch）更新（`ppo2/ppo2.py#L157-L166`）  
**（代码层面的优化）**

在每次 LEARNING 阶段，PPO 会先打乱 `N*M` 条数据，然后分成若干 mini-batch 做梯度下降。

常见错误包括：  
1. 每次都用整个批次做一次更新，而不是拆分多次；  
2. 随机采样的方式抽取 mini-batch，但结果导致并非每条数据都被训练到。

---

#### (7) 优势函数的标准化（`ppo2/model.py#L139`）  
**（代码层面的优化）**

在计算好 GAE 之后，PPO 会先对优势函数做标准化：减去均值再除以标准差。并且，这是 **在每个 mini-batch** 上独立做的，而不是对整个批次一次性做标准化。

- Andrychowicz, et al. (2021) 研究发现对 **每个 mini-batch** 做标准化或对 **整个 batch** 做标准化，对最终表现差别不大。（decision C67）

---

#### (8) 裁剪的目标函数（`ppo2/model.py#L81-L86`）  
**（理论相关）**

这是原论文所提出的“clipped”策略梯度目标函数。  

- Engstrom, Ilyas, et al. (2020) 发现当其他实现细节相同时，PPO 的裁剪目标和 TRPO 那种 KL 限制下的目标表现差不多。  
- Andrychowicz, et al. (2021) 发现 PPO 的裁剪目标整体上优于 vanilla PG、V-trace、AWR、V-MPO 等。

---

#### (9) 值函数损失的裁剪（`ppo2/model.py#L68-L75`）  
**（代码层面的优化）**

PPO 中，值函数的损失也被裁剪了。形式为：

\[
L^V = \max\Big[(V_\theta - V_\text{targ})^2,\big(\text{clip}(V_\theta, V_{\theta_{\mathrm{old}}} - \epsilon, V_{\theta_{\mathrm{old}}}+ \epsilon) - V_\text{targ}\big)^2\Big]
\]

- Engstrom, Ilyas, et al. (2020) 指出这种值函数损失裁剪没带来显着好处。  
- Andrychowicz, et al. (2021) 甚至发现它可能会带来负面影响（decision C13）。  
- 但是为了复现，我们还是实现了它。

---

#### (10) 总体损失和熵奖励（`ppo2/model.py#L91`）  
**（理论相关）**

PPO 的最终损失为 `loss = policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient`，其中熵项是为了鼓励探索。策略网络和价值网络共用一个优化器一起更新。

- Mnih 等人曾报告熵奖励可以改善探索（让策略分布保持一定随机性）。  
- Andrychowicz, et al. (2021) 发现这个熵奖励在连续控制环境里帮助不大（decision C13）。

---

#### (11) 全局梯度裁剪（`ppo2/model.py#L102-L108`）  
**（代码层面的优化）**

每次更新迭代时，会对策略和价值网络的全部梯度做一次全局的 L2 范数裁剪，限制不超过 0.5。

- Andrychowicz, et al. (2021) 发现这会带来一点小的正向增益（decision C68）。

---

#### (12) 调试变量（`ppo2/model.py#L115-L116`）  

PPO 代码里会输出一些调试用的变量：  
- `policy_loss`：所有数据上的平均策略损失  
- `value_loss`：所有数据上的平均价值损失  
- `entropy_loss`：所有数据上的平均熵值  
- `clipfrac`：有多少比例的数据触发了裁剪  
- `approxkl`：近似 KL 散度的值，具体实现对应 Schulman 博客中提到的 k1 估计量，即 `(-logratio).mean()`。还有另一种 `(ratio - 1) - logratio).mean()` 估计量，可做对比。

---

#### (13) 策略与价值共网或分网（`common/policies.py#L156-L160, baselines/common/models.py#L75-L103`）  
**（神经网络相关，代码层面的优化）**

PPO 默认使用一个两层 MLP（隐藏层各 64 个神经元，激活函数为 Tanh），然后再分出策略头和价值头（共享前面的网络）。伪代码如下：

```
network = Sequential(
    layer_init(Linear(obs_dim, 64)), Tanh(),
    layer_init(Linear(64, 64)), Tanh(),
)
value_head = layer_init(Linear(64, 1), std=1.0)
policy_head = layer_init(Linear(64, action_dim), std=0.01)

hidden = network(observation)
value = value_head(hidden)
action = Categorical(policy_head(hidden)).sample()
```

如果把 `value_network='copy'` 打开，则会分成两个完全独立的 MLP，一个专门输出价值，一个专门输出策略。  
我们在自己的代码里先实现了“分网”版本，然后改 10 行左右让其变成“共网”版本，方便对比。结果发现对于简单环境，“分网”往往效果更好，因为共网时策略和价值目标会相互竞争。

---

我们把上面前 12 条细节加上“分网”结构做进了一个 `ppo.py` 单文件实现，只有 322 行。然后改成共享网络，又做了一个 `ppo_shared.py`（317 行）。下图展示了它们的差异（略）。

在经典控制任务（如 CartPole 等）上的实验表明分网更容易收敛。当然，共网是官方 PPO 在 Atari 上的默认设置。因为在更复杂的环境里，共网也更节省计算量。

---

### 2. 9 个 Atari 专用实现细节

下面介绍与 Atari 环境紧密相关的 9 个实现细节。我们也提供了一个相应的逐行 PyTorch 视频教程。  

---

#### (1) `NoopResetEnv`（`common/atari_wrappers.py#L12`）  
**（环境预处理）**

该 wrapper 在环境 `reset()` 的时候随机执行 1~30 次 no-op（空操作）。由此给智能体提供带随机性的初始状态。  
- 这种做法最早来自 (Mnih et al., 2015) 以及 (Machado et al., 2018)，是一种向环境注入随机性的方法。

---

#### (2) `MaxAndSkipEnv`（`common/atari_wrappers.py#L97`）  
**（环境预处理）**

该 wrapper 每次执行动作时，会“跳帧”4 次（即重复上一个动作 4 帧），并在这 4 帧期间把奖励累加返回。它还会取最近两帧画面的逐像素最大值，以解决 Atari 某些交替闪烁帧的问题。

- 这个设计来自 (Mnih et al., 2015)，可以大幅加速训练，因为环境步进通常比神经网络推理要便宜得多。
- 取帧最大值是为了消除一些游戏中因硬件限制导致的交替闪烁（flickering）。

---

#### (3) `EpisodicLifeEnv`（`common/atari_wrappers.py#L61`）  
**（环境预处理）**

在具有“生命数（life）”的 Atari 游戏（如 Breakout）里，该 wrapper 会把每次失去一条生命都视为一次新的 episode，令 `done=True`。  
- 这是 (Mnih et al., 2015) 提出的做法，以加速训练。  
- 不过 (Bellemare et al., 2016) 提出有时这反而会降低性能，(Machado et al., 2018) 也建议别用它。可见这是一个有争议的做法，但官方 PPO 里默认采用了。

---

#### (4) `FireResetEnv`（`common/atari_wrappers.py#L41`）  
**（环境预处理）**

对于那些必须执行 FIRE 动作才能开始的游戏（比如 Breakout），该 wrapper 会在 `reset()` 后自动执行一次 FIRE 动作，让游戏开始。  
- 来源不明。有人在 `openai/baselines#240` 里说 DeepMind 和 OpenAI 内部都说不清它最初是谁加的。

---

#### (5) `WarpFrame`（图像变换，`common/atari_wrappers.py#L134`）  
**（环境预处理）**

该 wrapper 将原始 210×160 的 RGB 图像取其 Y（亮度）通道并缩放到 84×84。  
- 来源 (Mnih et al., 2015)，因为灰度图加缩放是 Atari 常见的输入预处理做法。  
- 我们在自己的实现中直接用 `gym.wrappers.ResizeObservation` 和 `gym.wrappers.GrayScaleObservation` 实现了同样的效果。

---

#### (6) `ClipRewardEnv`（`common/atari_wrappers.py#L125`）  
**（环境预处理）**

将奖励裁剪到 +1、0、-1 三种离散值。  
- 来源 (Mnih et al., 2015)，目的是让不同规模的分数在训练中有统一尺度。但这也意味着不再区分奖励幅度大小。

---

#### (7) `FrameStack`（`common/atari_wrappers.py#L188`）  
**（环境预处理）**

在返回给智能体的观测中，累积最近 m 帧图像作为一个堆叠，通常设 m=4。这样智能体可以从静态帧推断物体运动方向和速度。  
- 来源 (Mnih et al., 2015)，“见到的 m 帧拼在一起当 Q 网络的输入” 的做法。

---

#### (8) 策略与价值共享的 Nature-CNN 网络（`common/policies.py#L157, common/models.py#L15-L26`）  
**（神经网络）**

在 Atari 上，PPO 用 (Mnih et al., 2015) 提出的三层卷积网络做特征提取，再接全连接层得到一个 512 维隐层，然后分别接一个策略头（输出离散动作）和一个价值头（输出标量）。伪代码：

```
hidden = Sequential(
    layer_init(Conv2d(4, 32, 8, stride=4)), ReLU(),
    layer_init(Conv2d(32, 64, 4, stride=2)), ReLU(),
    layer_init(Conv2d(64, 64, 3, stride=1)), ReLU(),
    Flatten(),
    layer_init(Linear(64 * 7 * 7, 512)), ReLU(),
)
policy = layer_init(Linear(512, action_dim), std=0.01)
value = layer_init(Linear(512, 1), std=1)
```

这样可以节省计算，相比策略网和价值网完全分开时更快。

---

#### (9) 图像归一化到 [0, 1]（`common/models.py#L19`）  
**（环境预处理）**

对输入图像做 “/255.0” 的归一化，将像素值从 0~255 缩放到 0~1 之间。  
- 实践经验表明，如果不做这一步，在初次更新时 KL 散度会迅速爆炸，因为初始权重的尺度不匹配。

---

我们在 `ppo.py` 的基础上增加约 40 行代码，把以上 9 条 Atari 细节加进去，形成一个单文件版本 `ppo_atari.py`（339 行）。下图展示了二者 diff。

然后使用官方超参（在 `baselines/ppo2/defaults.py` 里）：

```
def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )
```

其中：
- `nsteps=128`（每个环境 rollout 128 步后再学习一次）  
- `nminibatches=4`（把 128*N 条数据分成 4 个 mini-batch）  
- `lam=0.95`（GAE 的 λ）  
- `gamma=0.99`（折扣因子）  
- `noptepochs=4`（每次更新迭代时，对整个数据重复训练 4 个 epoch）  
- `ent_coef=0.01`（熵系数）  
- `lr=lambda f: 2.5e-4 * f`（线性退火）  
- `cliprange=0.1`（PPO 的裁剪区间 ε）

另外，论文里说并行环境数 N=8（“8 个 actor”），而 `common/cmd_util.py#L167` 里是根据 CPU 个数动态设定。我们就显式改成 N=8。

最后在 Atari 上做实验，结果与原实现（在 Breakout 等环境）非常接近。

---

### 3. 连续动作域（MuJoCo 等）的 9 条细节

对于机器人学等连续动作环境，PPO 有一些专门的实现。我们亦准备了对应的视频教程（其中跳过了第 4 条实现细节），因此视频标题里写的是“8 Details for Continuous Actions”。

---

#### (1) 连续动作通过正态分布采样（`common/distributions.py#L103-L104`）  
**（理论）**

在连续动作空间里，常见做法是让策略输出某个多维正态分布 \(\mathcal{N}(\mu,\sigma^2)\)，然后从中采样。  
- 这是 (Schulman et al., 2015)、(Duan et al., 2016) 等早期工作的标准做法。

---

#### (2) 与状态无关的对数标准差（`common/distributions.py#L104`）  
**（理论）**

网络输出动作均值 \(\mu\)，但标准差 \(\sigma\) 则是一个独立的可训练参数（log-std）。并且缺省初始化为 0（对应 \(\sigma=1\)）。  
- 也有工作(如 Haarnoja et al., 2018) 使用状态相关的标准差。Andrychowicz, et al. (2021) 实验表明这二者差异并不大（决策 C59）。

---

#### (3) 各维动作独立（`common/distributions.py#L238-L246`）  
**（理论）**

对多维动作（如 [a1, a2, ...]），PPO 默认将它们视为独立分布的乘积，即 \( p(a)=\prod_i p(a_i)\)。  
- 这是一种简化假设，实践中或许有些任务在不同维度动作上存在相关性，然而要建模全协方差矩阵比较复杂。也有自回归（auto-regressive）等方法 (Metz et al., 2019; Zhang et al., 2019)，尚待更多验证。

---

#### (4) 策略网络与价值网络分开（`common/policies.py#L160, baselines/common/models.py#L75-L103`）  
**（神经网络）**

与 Atari 不同，MuJoCo 等连续控制任务里，PPO 默认使用两套相同结构的 MLP：  
- 都是 2×64 的 Tanh 隐藏层  
- 一套输出价值  
- 一套输出动作均值（另有一个可训练的 log-std）  

Andrychowicz, et al. (2021) 认为分网更好（决策 C47）。

---

#### (5) 动作截断到合法区间并存储原动作（`common/cmd_util.py#L99-L100`）  
**（代码层面的优化）**

在一些环境，动作有上下界 [low, high]。PPO 会在采样到动作后先截断到合法区间再执行，但保存数据时依然记录“截断前”的动作。  
- 这一做法可追溯到 (Duan et al., 2016) 等，但也有人用 \(\tanh\) 对动作进行 squashing，以天然限制范围 (Haarnoja et al., 2018)。后者可能效果更好 (Andrychowicz, et al., 2021, 决策 C63)。也有人指出截断法会带来偏差 (Chou, 2017; Fujita et al., 2018)。

---

#### (6) 观测归一化（`common/vec_env/vec_normalize.py#L4`）  
**（环境预处理）**

`VecNormalize` 对每个 timestep 的观测做减均值、除方差的处理（运行时维护一个滑动平均）。  
- 在连续控制里，这是非常常见的做法 (Duan et al., 2016)。  
- Andrychowicz, et al. (2021) 表明它对性能影响非常正面 (决策 C64)。

---

#### (7) 观测截断（`common/vec_env/vec_normalize.py#L39`）  
**（环境预处理）**

对归一化后的观测再进一步截断到 [-10, 10]。  
- Andrychowicz, et al. (2021) 发现当已经做了归一化后再截断帮助不大 (决策 C65)。但官方就这么实现，我们为了复现也照做。

---

#### (8) 奖励缩放（`common/vec_env/vec_normalize.py#L28`）  
**（环境预处理）**

`VecNormalize` 还会对奖励进行折扣加权式的标准化 (running discounted sum)，并除以其标准差，从而实现一个自适应的奖励缩放。  
- Engstrom, Ilyas, et al. (2020) 指出奖励缩放对最终训练表现有显着影响，并推荐使用。

---

#### (9) 奖励截断（`common/vec_env/vec_normalize.py#L32`）  
**（环境预处理）**

对缩放后的奖励进一步截断到 [-10, 10]。  
- 类似 (Mnih et al., 2015) 在 Atari 上对奖励裁剪的做法。是否真的在连续环境里有益，还缺乏明确定论。

---

我们在 `ppo.py` 基础上增加了约 25 行代码，把上述 9 条细节加进去，形成单文件 `ppo_continuous_action.py`（331 行）。接着用官方超参：

```
def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )
```

值得注意的是：
- `value_network='copy'` 就是分网模式  
- `num_envs=1`（在 `common/cmd_util.py#L167` 里默认这么设）  

最后我们在 MuJoCo 上做基准测试，结果和原实现极为接近。

---

### 4. LSTM 的 5 个实现细节

接下来介绍 PPO 中关于 LSTM 的 5 个细节。

---

#### (1) LSTM 层的权重初始化（`a2c/utils.py#L84-L86`）  
**（神经网络）**

LSTM 的各层权重用 `std=1` 初始化，偏置为 0。

---

#### (2) LSTM 状态初始化为 0（`common/models.py#L179`）  
**（神经网络）**

在推理和训练时，默认将隐藏状态和细胞状态都设为全 0。

---

#### (3) 回合结束时重置 LSTM 状态（`common/models.py#L141`）  
**（理论）**

如果 `done=True`，就说明一个回合结束，需要把 LSTM 的隐藏状态也重置为 0。

---

#### (4) 在小批次中保持序列一致（`a2c/utils.py#L81`）  
**（理论）**

在非 LSTM 的情形下，我们通常可以随机打乱数据来组建 mini-batch。但在 LSTM 时，数据顺序很重要。因此要基于子环境的时间序列来做小批次，而不是随意打乱。这样才能让隐藏状态在前后步骤间传递。

---

#### (5) 在训练时重建 LSTM 状态（`a2c/utils.py#L81`）  
**（理论）**

也就是说，在开始训练每个 mini-batch 时，要把在 rollout 阶段保存的初始 LSTM 状态恢复，然后让它在序列中前向传播，才能得到与采样时一致的隐藏状态，从而算出正确的 log-prob 等信息。

---

我们基于 `ppo_atari.py` 修改了约 60 行来支持这 5 条，变成了 `ppo_atari_lstm.py`（385 行）。然后在 Atari 上做实验（帧堆叠改成 1）。结果同样和原版接近。

---

### 5. MultiDiscrete 动作空间

MultiDiscrete 指动作由多个离散维度组成，如 `[a1, a2]` 可能分别表示“上下左右”按键和“AB 按钮”是否按下等。Gym 文档中是这么描述的：

```
class MultiDiscrete(Space):
    """
    - MultiDiscrete 动作空间由一系列离散子动作空间组成，每个都有各自的动作数量
    - 这在需要描述手柄或键盘输入时很有用
    ...
    """
```

PPO 要做的特殊处理就是：

---

#### (1) 各离散分量独立（`common/distributions.py#L215-L220`）  
**（理论）**

如果动作是 `[a1, a2, ..., aK]`，则 PPO 会将其视为独立的离散分布分量，相乘得到整体概率。  
- AlphaStar（Vinyals et al., 2019）和 OpenAI Five（Berner et al., 2019）都使用了 MultiDiscrete 动作空间。比如 OpenAI Five 的动作空间相当于 `MultiDiscrete([30, 4, 189, 81])`，有上百万种组合。

---

我们对 `ppo_atari.py` 进行了约 36 行修改，以支持 MultiDiscrete，做成 `ppo_multidiscrete.py`（335 行）。  
然后在 Gym-μRTS 环境（Huang et al., 2021）测试，表现与原版相符。

---

### 额外的 4 个实现细节

下面再提 4 个官方 PPO 未用、但可能在某些场景有用的技巧：

1. **Clip Range 退火（`ppo2/ppo2.py#L137`）**  
   - 类似学习率退火，也可以让 PPO 的裁剪范围 ε 随时间衰减。  
   - 在 openai/baselines 中其实有，但默认关掉。

2. **并行化梯度更新（`ppo2/model.py#L131`）**  
   - 在 `ppo1` 里可以并行地计算梯度再合并，以提速。但 `ppo2` 默认没启用。

3. **提早停止（early stopping）（`ppo/ppo.py#L269-L271`）**  
   - 这是 Schulman 在 modular_rl 以及 spinningup 中的一个做法：如果某次更新时估计到的 KL 超过阈值，就提前结束该次训练 epoch。可视为对稳定性的另一种保障。  
   - Dossa et al. 研究过这个技巧，发现可用于替代调大 epoch 数。  
   - 注意 spinningup 里只停止策略更新，我们的实现是策略和价值同时停。

4. **无效动作屏蔽（invalid action masking）(Vinyals et al., 2017; Huang and Ontañón, 2020)**  
   - 在 AlphaStar、OpenAI Five 等大型项目里，通过给不合法的动作在 logits 上设置负无穷，让智能体永远不会选到它们。  
   - 这能极大减少策略的搜索空间。对某些博弈尤为重要，比如微观操作多样的 RTS 游戏 (Huang et al., 2021)。  
   - 我们也在 `ppo_multidiscrete.py` 上加了约 30 行，做成了 `ppo_multidiscrete_mask.py`，测试表明能提升 Gym-μRTS 上的效果。

---

## 实验结果

如前文各节所示，我们的实现在各环境上都与官方版本非常吻合。在其他指标（比如策略损失、价值损失）上也吻合。可以点击查看一些我们用 Weights & Biases 做的可视化报表链接。

---

## 建议

在复现过程中，我们发现了一些实用的调试方法：

1. **完全固定随机种子**：然后逐步比对与参考实现输出是否一致（比如比较观测、动作采样、value、loss 等）。
2. **检查第一轮更新时 ratio 是否等于 1**：在第一轮更新的第一批 mini-batch 里，新旧策略本应相同，ratio 理论上应为 1。如果不是，则说明重建概率分布有问题。  
3. **监控 KL 散度**：如果 KL 跑得太高，可能策略更新过快或有 bug。  
4. **查看其他指标曲线**：比如 policy_loss, value_loss, entropy, clipfrac 等，与参考实现差别大就要排查。  
5. **Breakout 到 400 分** 的经验：如果你的 PPO 无法在 Breakout 训练到 400 分左右，很可能遗漏了一些关键实现细节。

如果你在做基于 PPO 的研究，以下建议有助于提高复现性：

- **枚举所用的实现细节**：像这样列清单，一条条写。  
- **开源并锁定源码**：用 Poetry、pipenv 或者 Docker 来固定依赖版本，避免“装完库就跑不起来”的情况。  
- **使用实验管理工具**：如 Weights & Biases、Neptune（商用），或 Aim、ClearML、Polyaxon（开源），可省下大量重复绘图的时间，并能更方便地管理超参、日志等。  
- **考虑单文件实现**：如果你需要灵活地修改 PPO 细节，单文件实现非常直观易懂，也方便做差异对比（git diff）来准确定位性能改进的来源。但缺点是代码可能难以复用、缺少面向对象封装。

---

## 讨论

### RL 库里的模块化设计是否好？

模块化设计对于软件工程是好事，但在 RL/ML 里可能分散了实现细节，导致初学者难以一次性看全。例如 Baselines 里 PPO 的逻辑分散在 `runner.py, model.py, cmd_util.py, policies.py, distributions.py, ...` 等多文件中，很难一眼掌握。  
我们并不是反对模块化，但希望社区能在保持库易用性的同时，尽量做好文档、代码注释或示例，帮助使用者理解实现细节。而对于做算法研究的人来说，可能单文件实现更好。

### 异步 PPO 是否更好？

不一定。异步版的 PPO（APPO, Berner et al., 2019）确实可以提高吞吐量，但会带来 stale experiences（经验滞后）等问题，目前也缺乏足够多环境的横向基准来证明它比同步 PPO 更好。

相比之下，一个简单且有效的做法是：**把环境本身提速**。例如在 C++ 里实现并行的向量化环境，或者在 GPU 上做物理仿真。我们演示了在 Atari 用 EnvPool 加速同步 PPO，就能比原版快 3 倍，还能在 Pong 上 5 分钟解决。

---

## 用 EnvPool 在 5 分钟内解决 Pong

EnvPool 是个新项目，用 C++ 和线程池加速了 Atari 的环境模拟。我们只需在 `ppo_atari.py` 基础上再改 60 行，把 Gym 替换为 EnvPool，就做成了 `ppo_atari_envpool.py`（365 行）。  
我们在 24 CPU + 1 块 RTX 2060 的机器上测得，同样是 8 个并行环境，EnvPool + PPO 比原版要快 3 倍左右，而且性能曲线无明显损失。

如再做一些超参微调，可以 5 分钟内用 PPO 训练出在 Pong 上打满分（21 分）的智能体。对比：  
- PARL 项目用单机 32 CPU + 1 块 P40，大约 10 分钟  
- RLlib 文档里提到 32~128 CPU 花 3 分钟  
- SeedRL 用 8 个 TPU v3 + 610 个 actor 花 45 分钟  
可见一个高效的同步实现并不比异步 IMPALA 差。

---

## 后续可研究的话题

1. **替代性的实现决策**：上文我们提到许多实现细节有些武断，或许可尝试不同变体。比如 Atari 的不同预处理管线 (Machado et al., 2018)，连续动作里不同的分布（Beta、全协方差 Gaussian、tanh squash 等），LSTM 初始化、或者分网时如何进一步分离价值等。  
2. **把向量化架构应用到有经验回放的算法**：DQN、SAC 等方法多半用单环境和经验回放。是否能在多环境并行收集的基础上减少对回放的需求，或提高训练效率？  
3. **价值函数优化的改进**：比如分离更新（Cobbe et al., 2021），或借鉴优先级经验回放 (Schaul et al.)。  
4. 其他方向不再赘述。

---

## 总结

复现 PPO 并非易事，即便有官方源码可参考。因为在过去几年里，官方源码几经修改，且其核心细节分散于各处，文档并不充分。与此同时，一些论文（Engstrom, Ilyas, et al. 2020；Andrychowicz, et al. 2021）对实现细节做了消融分析，但并未做成系统的教程，也主要聚焦在机器人学任务。本博客因此做了一个回溯原点、面面俱到的工作：

- 系统梳理了官方 PPO 的**所有**关键实现细节（共 37 条），不论是 Atari、MuJoCo、LSTM，还是 MultiDiscrete；  
- 逐一在 PyTorch 中做了单文件从零实现；  
- 展示了与原始实现高度一致的实验结果；  
- 探讨了软件工程上的启示，以及如何借助更快的环境实现加速（EnvPool）。  

希望这能帮助更多人快速上手 PPO，并在此基础上自由地定制或做新的研究。

---

## 致谢

感谢 Weights and Biases 提供的免费学术许可证，帮助我们跟踪实验进度。作者 Shengyi 也想感谢 Angelica Pan、Scott Condron、Ivan Goncharov、Morgan McGuire、Jeremy Salwen、Cayla Sharp、Lavanya Shukla 和 Aakarshan Chauhan 在视频制作过程中的支持。

---

## 参考文献

（以下为原文列出的参考文献，保持英文不变，请读者注意查看。）

- Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. 2017 Jul 20.  
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.  
- Engstrom L, Ilyas A, Santurkar S, Tsipras D, Janoos F, Rudolph L, Madry A. Implementation matters in deep policy gradients: A case study on ppo and trpo. International Conference on Learning Representations, 2020.  
- Andrychowicz M, Raichuk A, Stańczyk P, Orsini M, Girgin S, Marinier R, Hussenot L, Geist M, Pietquin O, Michalski M, Gelly S. What matters in on-policy reinforcement learning? a large-scale empirical study. International Conference on Learning Representations, 2021.  
- Mnih V, Kavukcuoglu K, Silver D, Rusu AA, Veness J, Bellemare MG, Graves A, Riedmiller M, Fidjeland AK, Ostrovski G, Petersen S. Human-level control through deep reinforcement learning. nature. 2015 Feb;518(7540):529-33.  
- Machado MC, Bellemare MG, Talvitie E, Veness J, Hausknecht M, Bowling M. Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents. Journal of Artificial Intelligence Research. 2018 Mar 19;61:523-62.  
- Schulman J, Levine S, Abbeel P, Jordan M, Moritz P. Trust region policy optimization. In International conference on machine learning 2015 Jun 1 (pp. 1889-1897). PMLR.  
- Duan Y, Chen X, Houthooft R, Schulman J, Abbeel P. Benchmarking deep reinforcement learning for continuous control. In International conference on machine learning 2016 Jun 11 (pp. 1329-1338). PMLR.  
- Haarnoja T, Zhou A, Abbeel P, Levine S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning 2018 Jul 3 (pp. 1861-1870). PMLR.  
- Chou PW. The beta policy for continuous control reinforcement learning (Doctoral dissertation, Master’s thesis. Pittsburgh: Carnegie Mellon University). 2017.  
- Fujita Y, Maeda SI. Clipped action policy gradient. In International Conference on Machine Learning 2018 Jul 3 (pp. 1597-1606). PMLR.  
- Bellemare M, Srinivasan S, Ostrovski G, Schaul T, Saxton D, Munos R. Unifying count-based exploration and intrinsic motivation. Advances in neural information processing systems. 2016;29:1471-9.  
- Tavakoli A, Pardo F, Kormushev P. Action branching architectures for deep reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence 2018 Apr 29 (Vol. 32, No. 1).  
- Metz L, Ibarz J, Jaitly N, Davidson J. Discrete sequential prediction of continuous actions for deep rl. arXiv preprint arXiv:1705.05035. 2017 May 14.  
- Zhang Y, Vuong QH, Song K, Gong XY, Ross KW. Efficient entropy for policy gradient with multidimensional action space. arXiv preprint arXiv:1806.00589. 2018 Jun 2.  
- Huang S, Ontañón S. A closer look at invalid action masking in policy gradient algorithms. arXiv preprint arXiv:2006.14171. 2020 Jun 25.  
- Huang, S., Ontan’on, S., Bamford, C., & Grela, L. Gym-μRTS: Toward Affordable Full Game Real-time Strategy Games Research with Deep Reinforcement Learning. In Proceedings of the 2021 IEEE Conference on Games (CoG).  
- Vinyals O, Babuschkin I, Czarnecki WM, Mathieu M, Dudzik A, Chung J, Choi DH, Powell R, Ewalds T, Georgiev P, Oh J. Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature. 2019 Nov;575(7782):350-4.  
- Berner C, Brockman G, Chan B, Cheung V, Dębiak P, Dennison C, Farhi D, Fischer Q, Hashme S, Hesse C, Józefowicz R. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680. 2019 Dec 13.  
- Vinyals O, Ewalds T, Bartunov S, Georgiev P, Vezhnevets AS, Yeo M, Makhzani A, Küttler H, Agapiou J, Schrittwieser J, Quan J. Starcraft ii: A new challenge for reinforcement learning. arXiv preprint arXiv:1708.04782. 2017 Aug 16.  
- Dossa RF, Huang S, Ontañón S, Matsubara T. An Empirical Investigation of Early Stopping Optimizations in Proximal Policy Optimization. IEEE Access. 2021 Aug 23;9:117981-92.  
- Espeholt L, Soyer H, Munos R, Simonyan K, Mnih V, Ward T, Doron Y, Firoiu V, Harley T, Dunning I, Legg S. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In International Conference on Machine Learning 2018 Jul 3 (pp. 1407-1416). PMLR.  
- Petrenko A, Huang Z, Kumar T, Sukhatme G, Koltun V. Sample factory: Egocentric 3d control from pixels at 100000 fps with asynchronous reinforcement learning. In International Conference on Machine Learning 2020 Nov 21 (pp. 7652-7662). PMLR.  
- Makoviychuk V, Wawrzyniak L, Guo Y, Lu M, Storey K, Macklin M, Hoeller D, Rudin N, Allshire A, Handa A, State G. Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning. ArXiv, abs/2108.10470. 2021.  
- Cobbe K, Hesse C, Hilton J, Schulman J. Leveraging procedural generation to benchmark reinforcement learning. In International conference on machine learning 2020 (pp. 2048-2056). PMLR.  
- Terry JK, Black B, Hari A, Santos L, Dieffendahl C, Williams NL, Lokesh Y, Horsch C, Ravi P. Pettingzoo: Gym for multi-agent reinforcement learning. Advances in Neural Information Processing Systems, 34, 2021.

---

## 附录

在此补充一个关于 `procgen` 环境的实现细节：

- **(1) 基于 IMPALA 风格的 CNN（`common/models.py#L28`）**：在 `openai/train-procgen` 仓库里，官方默认用的是类似 IMPALA (Espeholt et al., 2018) 的卷积网络结构（不带 LSTM）。  
- 我们做了 `ppo_procgen.py`，与 `ppo_atari.py` 相比增加约 60 行来适配此网络，并在 `ProcgenEnv` 上跑了 2500 万步 (distribution_mode="easy")。同样结果与官方相近。

--
