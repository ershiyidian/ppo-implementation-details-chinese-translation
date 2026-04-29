# PPO 实现细节中文翻译

这是 ICLR Blog Track 文章 **The 37 Implementation Details of Proximal Policy Optimization** 的非官方中文学习翻译与阅读辅助仓库。

## 原文信息

- 原文标题：[The 37 Implementation Details of Proximal Policy Optimization][original]
- 原文作者：Huang, Shengyi; Dossa, Rousslan Fernand Julien; Raffin, Antonin; Kanervisto, Anssi; Wang, Weixun
- 发布时间：2022-03-25
- 原文平台：ICLR Blog Track
- 原文代码仓库：[vwxyzjn/ppo-implementation-details][code]
- 原文实验记录：[Weights & Biases - ppo-details][wandb]

## 仓库内容

```text
.
├── README.md              # 仓库说明
├── trans.md               # 中文译文与对照说明
└── media_index.md         # 原文图片、视频和交互式实验面板索引
```

## 阅读入口

- 中文译文：[trans.md](trans.md)
- 原文媒体索引：[media_index.md](media_index.md)
- 原文页面：[ICLR Blog Track][original]

## 翻译原则

- 不省略原文中的关键论点、实现细节类别、实验结论和引用线索。
- 对 PPO、GAE、VecEnv、MultiDiscrete、LSTM、EnvPool 等术语保持一致翻译。
- 保留必要英文术语，避免为了中文流畅而牺牲技术准确性。
- 将原文图片、视频和 W&B 交互面板作为外链索引列出，方便读者对照原始材料。
- 对不确定或依赖原文上下文的内容，以原文为准。

## 版权与使用边界

原文页面和其中的图片、视频、交互面板、代码链接等材料归原作者和对应平台所有。本仓库仅作为个人学习、课程阅读和技术理解用途维护，不声称拥有原文版权。

由于原文所在站点保留版权，本仓库不会将原文页面的全部图片、视频或交互式内容复制到仓库内；相关材料以外链索引方式引用。若原作者或版权方认为本仓库内容不适合公开，请通过 Issue 联系处理。

## 引用

学术或正式场景请引用原文：

```bibtex
@inproceedings{huang2022the37,
  author = {Huang, Shengyi and Dossa, Rousslan Fernand Julien and Raffin, Antonin and Kanervisto, Anssi and Wang, Weixun},
  title = {The 37 Implementation Details of Proximal Policy Optimization},
  booktitle = {ICLR Blog Track},
  year = {2022},
  url = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/}
}
```

[original]: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
[code]: https://github.com/vwxyzjn/ppo-implementation-details
[wandb]: https://wandb.ai/vwxyzjn/ppo-details
