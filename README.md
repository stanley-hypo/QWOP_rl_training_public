# 强化学习 QWOP 训练环境

本项目包含 QWOP 模拟物理环境和 PPO 强化学习框架，用于探索 QWOP 游戏的极限。

如果您成功训练了模型，您可以把回放文件上传我们的网站冲榜 [run.lyihub.com](https://run.lyihub.com/)，详见**上传到我们的游戏榜单**部分。

## 仓库内容

- Win64 预编译游戏物理模拟器
- 基于 pybind11 的单环境和批量环境 Python API
- 经典 100 米、110 米跨栏和跳远模式
- 可修改、可重新编译的奖励函数
- PPO 训练和策略蒸馏脚本
- 导出回放脚本

## 环境要求

- 64 位 Windows 10 或 11
- 64 位 python；本项目使用 python 3.10
- Git
- CMake 3.20 或更高版本
- 支持 C++20 的 Windows C++ 编译器和 Windows SDK；本项目使用 MSVC 验证，如用其他编译器，需要修改 build.py
- [Microsoft Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)，用于运行仓库提供的物理 DLL

## 安装

```
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\build.py
.\.venv\Scripts\python.exe scripts\check_environment.py
```

上面的 PyTorch 命令以 CUDA 12.8 为例。显卡驱动不匹配时，请在
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 Windows、Pip 和自己的 CUDA 版本，
再执行生成的命令。

## 训练 PPO

```
.\.venv\Scripts\python.exe scripts\train_ppo.py --config configs\ppo_example.json
```

### 配置文件

`configs/ppo_example.json` 显式列出了全部可用参数。

PPO 参数：

| 参数 | 示例值 | 默认值 | 说明 |
| --- | ---: | ---: | --- |
| `num_envs` | `32` | `32` | 并行运行的环境数量，必须大于 0。数值越大，每次 update 收集的数据越多，占用的内存也越多。 |
| `num_threads` | `4` | `1` | C++ 批量环境使用的工作线程数，必须大于 0；实际线程数不会超过 `num_envs`。 |
| `rollout_steps` | `128` | `128` | 每个环境在一次 update 中连续采样的步数。 |
| `updates` | `100` | `10` | 本次命令执行的 PPO 外层更新次数；续训时该值表示在检查点基础上再执行多少次。 |
| `epochs` | `4` | `4` | 每批 rollout 数据被重复优化的轮数。 |
| `minibatches` | `4` | `4` | 将一次 update 的样本拆分为小批次进行优化时使用的目标分组数，必须大于 0 且不能超过该次 update 的样本总数。 |
| `gamma` | `0.99` | `0.99` | 奖励折扣因子；越接近 1，策略越重视远期奖励。 |
| `gae_lambda` | `0.95` | `0.95` | GAE 参数，用于控制 advantage 估计的偏差与方差。 |
| `clip_coef` | `0.2` | `0.2` | PPO 概率比裁剪范围，限制单次策略更新幅度。 |
| `ent_coef` | `0.01` | `0.01` | 熵奖励在总损失中的系数；增大后通常会鼓励更多探索。 |
| `vf_coef` | `0.5` | `0.5` | value loss 在总损失中的权重。 |
| `learning_rate` | `0.0003` | `0.0003` | actor-critic Adam 优化器的学习率。 |
| `seed` | `1` | `1` | PyTorch 和 NumPy 的随机种子。 |

环境与网络参数：

| 参数 | 示例值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `reward_profile` | `"classic_100m"` | `"classic_100m"` | 奖励方案，可选值为 `classic_100m`、`hurdle_110m`、`long_jump_30m`。具体规则见“奖励方案”。 |
| `game_mode` | `"classic_100m"` | `"auto"` | 游戏模式，可使用 `auto` 自动匹配奖励方案，也可显式指定同名模式；模式与奖励方案不匹配时直接报错。 |
| `hidden_sizes` | `[128, 128]` | `[128, 128]` | actor 和 critic 共享 MLP 的隐藏层宽度列表。JSON 使用整数数组，命令行使用逗号分隔格式，例如 `--hidden-sizes 256,256`。 |
| `frame_stack` | `1` | `1` | 允许模型观察前N帧，可以用于引入时间维度，必须大于 0；最终输入维度会乘以该值。episode 重置时会用初始观测填满所有帧。 |

输出与续训参数：

| 参数 | 示例值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `checkpoint` | `"checkpoints/ppo_example.pt"` | `"checkpoints/ppo_running.pt"` | 最终检查点的保存路径；周期检查点会在文件名后添加 `_update_XXXXXX`。 |
| `save_interval` | `100` | `100` | 每隔多少次 update 保存一个周期检查点；设为 0 或负数时关闭周期保存，但训练结束时仍会保存最终检查点。 |
| `resume_from` | `null` | `null` | 要恢复的检查点路径；`null` 表示从头训练。恢复后会继续累计 `absolute_update` 和 `global_step`。 |
| `reset_optimizer_on_resume` | `false` | `false` | 续训时是否丢弃检查点中的优化器状态。设为 `true` 时保留模型参数，但使用新的优化器状态和当前学习率。 |
| `run_dir` | `"runs"` | `"runs"` | TensorBoard 日志和训练元数据的根目录。 |
| `run_name` | `"ppo_example"` | `""` | 本次运行在 `run_dir` 下的子目录名；空字符串会自动使用当前时间生成名称。 |

RND 探索奖励参数：

| 参数 | 示例值 | 默认值 | 说明 |
| --- | ---: | ---: | --- |
| `rnd_beta` | `0.0` | `0.0` | RND 内在奖励的初始权重。只有该值大于 0 时才会创建并训练 RND 网络；`0` 表示完全关闭 RND。 |
| `rnd_beta_final` | `null` | `null` | RND 权重衰减或增长后的目标值；`null` 表示始终使用 `rnd_beta`。 |
| `rnd_beta_decay_updates` | `0` | `0` | 从 `rnd_beta` 线性变化到 `rnd_beta_final` 所需的 update 数；小于等于 0 时不进行变化。 |
| `rnd_learning_rate` | `0.0001` | `0.0001` | RND predictor Adam 优化器的学习率，仅在 RND 启用时生效。 |
| `rnd_hidden_size` | `128` | `128` | RND target 和 predictor 网络的隐藏层宽度，仅在 RND 启用时生效。 |
| `rnd_feature_dim` | `64` | `64` | RND 输出特征维度，仅在 RND 启用时生效。 |

模型检查点会写入 `checkpoints/`，TensorBoard 日志和训练记录会写入 `runs/`。

使用 tensorboard 查看训练指标：

```
.\.venv\Scripts\tensorboard.exe --logdir runs
```

### TensorBoard 指标

主要看两个值：

| 指标 | 含义 | 主要看法 |
| --- | --- | --- |
| `charts/final_distance_mean` | 当前 update 内所有已结束 episode 的最终前进距离均值。经典跑和跨栏使用赛道距离；跳远使用 30 米助跑距离加落点距离。 | 最直接的任务成绩指标，长期上升说明策略跑得更远。不同游戏模式之间不要直接横向比较。 |
| `charts/episode_return_mean` | 当前 update 内所有已结束 episode 的环境奖励总和均值，不包含 RND 内在奖励。 | 用于判断策略是否正在适应奖励函数，应结合最终距离一起看，避免只提高奖励却没有提高实际成绩。 |

### 奖励函数

奖励函数定义在 `src/reward_profiles.cpp`，每套奖励方案都对应同名的游戏模式。使用 `game_mode="auto"` 可以自动选择对应模式；如果显式指定了不匹配的模式，环境会直接报错。

以下是仓库自带的奖励函数：
- `classic_100m`：正向前进奖励累计上限为 100 分；摔倒或超过 120 秒时扣 10 分；通关时额外获得 `120 - 已用秒数` 分。
- `hurdle_110m`：正向前进奖励累计上限为 110 分；每个栏架的前脚、身体和后脚高度奖励上限分别为 8、5、8 分；跨栏犯规或摔倒时扣 10 分；通关时额外获得 `120 - 已用秒数` 分。
- `long_jump_30m`：30 米助跑阶段的正向前进奖励累计上限为 30 分；摔倒时扣 10 分；通关时获得跳远距离的 5 倍奖励；落地后向前走出沙坑的距离按 0.5 倍给予奖励。

修改奖励函数后需要重新构建 pyd 文件。

### 蒸馏

当你想更换模型的时候，你可以用新模型蒸馏旧模型：

```
.\.venv\Scripts\python.exe scripts\distill_policy.py `
  --teacher checkpoints\ppo_example.pt `
  --output checkpoints\ppo_classic_100m_student.pt `
  --student-hidden-sizes 128,128 `
  --num-envs 2048 `
  --num-threads 32 `
  --device auto
```

## 导出操作回放

### 导出确定性回放

下面从 checkpoint 执行一次确定性策略，并将操作序列写入 `.bin`：

```
.\.venv\Scripts\python.exe scripts\export_policy_replay.py `
  --checkpoint checkpoints\ppo_example.pt `
  --output replays\classic_100m.bin
```

脚本自动读取 checkpoint 中的网络宽度、堆帧数和游戏模式。默认每一步选择概率最大的动作，因此相同 checkpoint 会得到可复现的操作序列。

上述脚本会返回以下格式：

```
episode=1 steps=3600 score=98.000 distance=98.000 time=120.000 success=0
```

score 是游戏结束时的最终距离，这里代表走了98.0m，time 是消耗时间，这里代表花费了120s。

### 抽取随机最佳记录

批量搜索会并行执行多个环境，随机采样指定数量的完整 episode，最后只保存最佳操作序列：

```
.\.venv\Scripts\python.exe scripts\search_policy_best.py `
  --checkpoint checkpoints\ppo_example.pt `
  --output replays\classic_100m_best.bin `
  --episodes 4096 `
  --num-envs 4096 `
  --num-threads 128 `
  --device auto
```

经典 100 米和 110 米跨栏优先选择成功通关的 episode，再按用时从短到长排序；没有通关记录时选择前进距离最远的 episode。跳远优先选择成功走出沙坑的 episode，再按有效落点距离从大到小排序。`--episodes` 越大，找到好记录的机会越高，但随机采样结果不代表策略的理论极限。

## 上传到我们的游戏榜单

训练完成后，将操作回放导出为 .bin 文件，即可上传成绩至排行榜网站 [run.lyihub.com](https://run.lyihub.com/)。

使用仓库提供的 `running_physics.dll` 生成回放，才能保证与网站使用相同的物理模拟。自行替换 DLL 后生成的回放不作保证。

### 上传与验证流程：

打开排行榜页面，点击游戏画面下方的“上传”按钮，选择您导出的 .bin 回放文件。系统将自动完整播放该回放，回放录像通关后，右侧的“提交成绩”按钮将被激活，此时即可正式提交。

### 排名规则：

排行榜涵盖三个项目：100 米和跨栏以完成时间升序排列（用时越短排名越高），跳远以有效距离降序排列（距离越远排名越高）。每个榜单仅展示前 50 名。若同一玩家名再次提交更优成绩，系统将自动更新其个人最佳记录。

### 文件限制与常见问题：

回放文件最多支持 30000 个物理步。

若上传网页遇到提示“无效的 replay 文件”，请确认该文件由本仓库提供的回放导出脚本生成且未经任何修改。

若“提交成绩”始终为不可用状态，通常表示该回放未能成功完成对应比赛，请更换有效回放后重试。

## 许可

本仓库采用“开源训练框架 + 单独授权预编译物理后端”的许可结构。源码、训练脚本、示例配置和文档按 Apache License 2.0 授权，详见 [LICENSE](LICENSE)。

`bin/running_physics.dll` 不属于 Apache License 2.0，按单独的二进制许可分发，详见 [`bin/running_physics.LICENSE.txt`](bin/running_physics.LICENSE.txt)。

公开源码通过 [`src/physics_backend_api.h`](src/physics_backend_api.h) 和 [`docs/physics-backend-abi.md`](docs/physics-backend-abi.md) 描述运行时 ABI，允许用户独立实现替代物理后端。

第三方软件及相关声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢与第三方软件

感谢 QWOP 作者 Bennett Foddy。QWOP 名称及相关权利归 Bennett Foddy 或其权利人所有。本项目仅供学习、交流和研究使用，与原游戏作者无关联。

本项目使用或包含以下第三方软件：

- [pybind11](https://github.com/pybind/pybind11)
- [musl libc](https://musl.libc.org/)
- [Box2D](https://box2d.org/)
