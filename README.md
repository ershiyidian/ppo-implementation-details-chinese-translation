# PPO 实现细节中文翻译

> **原文标题**: [The 37 Implementation Details of Proximal Policy Optimization][original-link]  
> **原文链接**: <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=vectorized%20architecture>  
> **原文作者**: [Huang, Shengyi](https://github.com/vwxyzjn) 等

## 简介

这是对 **The 37 Implementation Details of Proximal Policy Optimization** 一文的非官方中文翻译，原文发布于 ICLR Blog Track。该文从多方面介绍了 PPO（Proximal Policy Optimization）算法的实现细节，涵盖 Atari、MuJoCo、LSTM、MultiDiscrete 动作空间等内容。

- 原文版权归原作者所有，本仓库中仅提供个人学习用途的译文，若有侵权请联系删除。
- 若想了解更多上下文或对原文中某些细节进行深入探讨，请参考[原文地址][original-link]。
- 本译文可能存在理解或表述不足之处，欢迎在 Issue 中提出改进建议。

## 内容概览

1. **为什么 PPO 难以复现**  
2. **官方实现代码谱系分析**  
3. **37 个实现细节的完整解读**  
   - 核心部分 (13 个)  
   - Atari 专用 (9 个)  
   - MuJoCo 连续动作 (9 个)  
   - LSTM (5 个)  
   - MultiDiscrete (1 个)  
4. **额外可能有用的 4 个细节**  
5. **实验结果与对比**  
6. **在 EnvPool 上的加速示例**  
7. **总结与后续工作**  

## 免责声明

- 本仓库是[“The 37 Implementation Details of Proximal Policy Optimization”][original-link] 的**非官方中文翻译**，仅供学习参考。
- 翻译内容可能与原文有出入，一切内容以原英文版为准。
- 原文版权及归属权均为原作者所有。若原作者对本译文的公开有任何异议，请通过本仓库的 issue 或邮件联系我。

---

[original-link]: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#:~:text=vectorized%20architecture
