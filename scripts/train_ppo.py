from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from running_env import VectorRunningGameEnv


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:
    raise SystemExit(
        "PyTorch and TensorBoard must be installed in this Python environment. "
        "Install them, then rerun: python scripts\\train_ppo.py"
    ) from exc


@dataclass
class PPOConfig:
    # PPO 的采样规模：每次 update 会收集 num_envs * rollout_steps 条转移样本。
    num_envs: int = 32
    rollout_steps: int = 128
    # updates 是训练外层循环次数；epochs/minibatches 控制同一批 rollout 数据被重复优化多少遍。
    updates: int = 10
    epochs: int = 4
    minibatches: int = 4
    # gamma 越接近 1，越重视远期奖励；gae_lambda 控制 advantage 估计的偏差/方差折中。
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # clip_coef 是 PPO 的核心保险丝：限制新旧策略概率比变化过大。
    clip_coef: float = 0.2
    # ent_coef 鼓励策略保持随机性；vf_coef 控制 value loss 在总 loss 中的权重。
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    seed: int = 1
    num_threads: int = 1
    checkpoint: Path = ROOT / "checkpoints" / "ppo_running.pt"
    resume_from: Path | None = None
    reset_optimizer_on_resume: bool = False
    reward_profile: str = "classic_100m"
    game_mode: str = "auto"
    run_dir: Path = ROOT / "runs"
    run_name: str = ""
    save_interval: int = 100
    hidden_sizes: tuple[int, ...] = (128, 128)
    frame_stack: int = 1
    rnd_beta: float = 0.0
    rnd_beta_final: float | None = None
    rnd_beta_decay_updates: int = 0
    rnd_learning_rate: float = 1e-4
    rnd_hidden_size: int = 128
    rnd_feature_dim: int = 64


def resolve_path(value: Path | str) -> Path:
    # 允许配置文件里写相对路径；统一解析到项目根目录下，避免运行目录改变后找不到文件。
    path = value if isinstance(value, Path) else Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    if "hidden_sizes" in data:
        data["hidden_sizes"] = tuple(int(value) for value in data["hidden_sizes"])
    return data


def make_config(args: argparse.Namespace) -> PPOConfig:
    # 配置优先级：PPOConfig 默认值 < JSON 配置文件 < 命令行参数。
    config_values = {}
    if args.config:
        config_values.update(load_config(resolve_path(args.config)))

    valid_fields = {field.name: field for field in fields(PPOConfig)}
    unknown = sorted(set(config_values) - set(valid_fields))
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")

    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config_values[key] = value

    for path_key in ("checkpoint", "resume_from", "run_dir"):
        if path_key in config_values:
            config_values[path_key] = None if config_values[path_key] is None else resolve_path(config_values[path_key])

    return PPOConfig(**config_values)


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise argparse.ArgumentTypeError("hidden sizes must contain at least one layer")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive")
    return sizes


class FrameStacker:
    def __init__(self, num_envs: int, base_obs_dim: int, frame_stack: int):
        if frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        self.num_envs = num_envs
        self.base_obs_dim = base_obs_dim
        self.frame_stack = frame_stack
        self.frames = np.zeros((num_envs, frame_stack, base_obs_dim), dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return self.base_obs_dim * self.frame_stack

    def reset(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        self.frames[...] = obs[:, None, :]
        return self.stacked()

    def update(self, obs: np.ndarray, done: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        done = np.asarray(done, dtype=bool)
        self.frames = np.roll(self.frames, shift=-1, axis=1)
        self.frames[:, -1, :] = obs
        if np.any(done):
            self.frames[done, :, :] = obs[done, None, :]
        return self.stacked()

    def stacked(self) -> np.ndarray:
        return self.frames.reshape(self.num_envs, self.obs_dim)


class ActorCritic(nn.Module):
    """共享前几层的 actor-critic 网络。

    actor 输出每个离散动作的 logits，用来形成策略 pi(a|s)；
    critic 输出当前状态的 value，用来估计“从这个状态往后还能拿多少回报”。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        # 这里的观测是向量特征，所以用 MLP；如果输入是图像，通常会换成 CNN。
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.Tanh())
            in_dim = hidden_size
        self.net = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, action_dim)
        self.critic = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor):
        # 返回 actor logits 和 critic value。value squeeze 到 [batch]，便于和 reward/return 对齐。
        hidden = self.net(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def act(self, obs: torch.Tensor):
        # 采样阶段使用当前策略随机选动作，并记录旧策略 log_prob。
        # 之后 PPO 更新会比较“新 log_prob”和这里保存的“旧 log_prob”。
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        # 训练阶段重新计算同一批动作在“当前新策略”下的 log_prob、entropy 和 value。
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class RNDModel(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 128, feature_dim: int = 64):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.target(obs)
        prediction = self.predictor(obs)
        return prediction, target

    def prediction_error(self, obs: torch.Tensor) -> torch.Tensor:
        prediction, target = self(obs)
        return ((prediction - target) ** 2).mean(dim=-1)


def rnd_beta_for_update(cfg: PPOConfig, update: int) -> float:
    if cfg.rnd_beta <= 0.0:
        return 0.0
    if cfg.rnd_beta_final is None or cfg.rnd_beta_decay_updates <= 0:
        return cfg.rnd_beta
    progress = min(1.0, max(0.0, update / float(cfg.rnd_beta_decay_updates)))
    return cfg.rnd_beta + (cfg.rnd_beta_final - cfg.rnd_beta) * progress


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: PPOConfig,
    update: int,
    global_step: int,
    rnd_model: RNDModel | None = None,
    rnd_optimizer: optim.Optimizer | None = None,
) -> None:
    # checkpoint 同时保存模型、优化器、配置和训练进度，方便断点续训。
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in cfg.__dict__.items()
        },
        "update": update,
        "global_step": global_step,
    }
    if rnd_model is not None:
        checkpoint["rnd_model"] = rnd_model.state_dict()
    if rnd_optimizer is not None:
        checkpoint["rnd_optimizer"] = rnd_optimizer.state_dict()
    torch.save(checkpoint, path)


def copy_linear_prefix(dst: torch.Tensor, src: torch.Tensor) -> None:
    rows = min(dst.shape[0], src.shape[0])
    if dst.ndim == 1:
        dst[:rows].copy_(src[:rows])
        return
    cols = min(dst.shape[1], src.shape[1])
    dst[:rows, :cols].copy_(src[:rows, :cols])


def load_model_compatible(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    for name, src in state_dict.items():
        if name not in current:
            continue
        dst = current[name]
        if dst.shape == src.shape:
            dst.copy_(src)
        elif name == "net.0.weight" and dst.ndim == 2 and src.ndim == 2 and dst.shape[1] > src.shape[1]:
            rows = min(dst.shape[0], src.shape[0])
            dst[:rows, -src.shape[1]:].copy_(src[:rows])
        elif name.startswith("net.") or name.startswith("actor.") or name.startswith("critic."):
            copy_linear_prefix(dst, src)
    model.load_state_dict(current)


def compute_gae(rewards, dones, values, next_value, cfg: PPOConfig):
    """用 Generalized Advantage Estimation 计算 advantage 和 return。

    delta 是一步 TD 误差：实际奖励 + 下一状态估值 - 当前状态估值。
    advantage 可以理解为“这个动作比 critic 原本预期的好多少”；
    return = advantage + value，是 critic 要回归的目标。
    """

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)
    # 从后往前递推，因为当前步的 advantage 依赖后续步的 TD 残差。
    for t in reversed(range(cfg.rollout_steps)):
        # done 后不再 bootstrap 下一状态价值，否则会把下一局/下一条 episode 的价值串进来。
        next_nonterminal = 1.0 - dones[t]
        next_values = next_value if t == cfg.rollout_steps - 1 else values[t + 1]
        delta = rewards[t] + cfg.gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class TrainingLogger:
    # TensorBoard 的薄封装：训练配置写成 text，指标写成 scalar。
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, step)

    def log(self, global_step: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, global_step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def train(cfg: PPOConfig) -> None:
    # 固定随机种子，让同样配置下的结果尽量可复现。
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 向量化环境一次并行跑多个游戏实例。PPO 是 on-policy 算法，吞吐越高越容易收集足够新鲜样本。
    env = VectorRunningGameEnv(
        cfg.num_envs,
        num_threads=cfg.num_threads,
        reward_profile=cfg.reward_profile,
        game_mode=cfg.game_mode,
    )
    cfg.game_mode = env.game_mode
    base_obs_np, _ = env.reset()
    stacker = FrameStacker(cfg.num_envs, base_obs_np.shape[1], cfg.frame_stack)
    obs_np = stacker.reset(base_obs_np)
    obs_dim = stacker.obs_dim
    action_dim = env.action_space_n

    model = ActorCritic(obs_dim, action_dim, cfg.hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    use_rnd = cfg.rnd_beta > 0.0
    rnd_model = RNDModel(obs_dim, cfg.rnd_hidden_size, cfg.rnd_feature_dim).to(device) if use_rnd else None
    rnd_optimizer = optim.Adam(rnd_model.predictor.parameters(), lr=cfg.rnd_learning_rate, eps=1e-5) if rnd_model else None
    resumed_update = 0
    global_step = 0
    if cfg.resume_from:
        # 续训时恢复模型和优化器状态；随后把学习率改成当前配置里的 learning_rate。
        checkpoint_data = torch.load(cfg.resume_from, map_location=device)
        load_model_compatible(model, checkpoint_data["model"])
        if "optimizer" in checkpoint_data and not cfg.reset_optimizer_on_resume:
            try:
                optimizer.load_state_dict(checkpoint_data["optimizer"])
            except ValueError:
                print("optimizer_state_skipped=1 reason=incompatible_model_shape")
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate
        if rnd_model and "rnd_model" in checkpoint_data:
            try:
                rnd_model.load_state_dict(checkpoint_data["rnd_model"])
            except RuntimeError:
                print("rnd_model_state_skipped=1 reason=incompatible_model_shape")
        if rnd_optimizer and "rnd_optimizer" in checkpoint_data and not cfg.reset_optimizer_on_resume:
            try:
                rnd_optimizer.load_state_dict(checkpoint_data["rnd_optimizer"])
            except ValueError:
                print("rnd_optimizer_state_skipped=1 reason=incompatible_model_shape")
            for group in rnd_optimizer.param_groups:
                group["lr"] = cfg.rnd_learning_rate
        resumed_update = int(checkpoint_data.get("update", 0))
        global_step = int(checkpoint_data.get("global_step", 0))
        print(
            f"resumed_from={cfg.resume_from} checkpoint_update={resumed_update} "
            f"global_step={global_step} reset_optimizer={int(cfg.reset_optimizer_on_resume)}"
        )
    run_name = cfg.run_name or datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    log_dir = cfg.run_dir / run_name
    logger = TrainingLogger(log_dir)
    # 把配置写进 TensorBoard，之后看曲线时能反查这次实验到底用了什么参数。
    logger.add_text("config/device", str(device), 0)
    logger.add_text("config/run_name", run_name, 0)
    for key, value in cfg.__dict__.items():
        logger.add_text(f"config/{key}", str(value), 0)

    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    # These buffers keep the same shape for the whole run. Reusing them avoids
    # repeated CUDA/Python allocator churn inside the hot update loop.
    obs_buf = torch.empty((cfg.rollout_steps, cfg.num_envs, obs_dim), device=device)
    actions_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device)
    rewards_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device)
    extrinsic_rewards_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device)
    intrinsic_rewards_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device)
    dones_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device)
    values_buf = torch.empty((cfg.rollout_steps, cfg.num_envs), device=device)
    batch_size = cfg.num_envs * cfg.rollout_steps
    minibatch_size = batch_size // cfg.minibatches
    if minibatch_size <= 0:
        raise ValueError("minibatch size must be positive")
    try:
        for update in range(1, cfg.updates + 1):
            absolute_update = resumed_update + update
            update_started = time.perf_counter()

            episode_finishes = 0
            episode_return_samples = []
            final_distance_samples = []
            rnd_beta = rnd_beta_for_update(cfg, update)
            # 第一阶段：用当前策略和环境交互，收集一批 on-policy 轨迹。
            for step in range(cfg.rollout_steps):
                global_step += cfg.num_envs
                obs_buf[step] = obs
                with torch.no_grad():
                    # 采样动作时不需要梯度；真正反传发生在 rollout 收集完之后。
                    actions, logprobs, _, values = model.act(obs)
                actions_np = actions.cpu().numpy()
                step_result = env.step_arrays(actions_np)
                base_next_obs_np = step_result["obs"]
                rewards_np = step_result["reward"]
                terminated_np = step_result["terminated"]
                truncated_np = step_result["truncated"]
                done_np = np.logical_or(terminated_np, truncated_np)
                next_obs_np = stacker.update(base_next_obs_np, done_np)
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
                extrinsic_reward = torch.tensor(rewards_np, dtype=torch.float32, device=device)
                if rnd_model:
                    with torch.no_grad():
                        intrinsic_reward = rnd_model.prediction_error(next_obs)
                        intrinsic_reward = intrinsic_reward / (intrinsic_reward.mean() + 1e-8)
                    total_reward = extrinsic_reward + rnd_beta * intrinsic_reward
                    intrinsic_rewards_buf[step] = intrinsic_reward
                else:
                    total_reward = extrinsic_reward
                    intrinsic_rewards_buf[step].zero_()
                # terminated 通常表示游戏自然结束；truncated 通常表示时间限制等外部截断。
                episode_returns += rewards_np.astype(np.float32)
                episode_finishes += int(done_np.sum())
                if np.any(step_result["has_final"]):
                    final_mask = step_result["has_final"]
                    episode_return_samples.extend(
                        float(episode_return)
                        for episode_return in episode_returns[final_mask]
                    )
                    final_distance_samples.extend(
                        float(distance)
                        for distance in step_result["final_distance"][final_mask]
                    )
                    episode_returns[final_mask] = 0.0

                actions_buf[step] = actions
                logprobs_buf[step] = logprobs
                rewards_buf[step] = total_reward
                extrinsic_rewards_buf[step] = extrinsic_reward
                dones_buf[step] = torch.tensor(done_np.astype(np.float32), dtype=torch.float32, device=device)
                values_buf[step] = values
                obs = next_obs

            with torch.no_grad():
                # rollout 最后一个 obs 还没有 value，需要 bootstrap 它来估计最后一步之后的未来。
                _, next_value = model(obs)
                advantages, returns = compute_gae(rewards_buf, dones_buf, values_buf, next_value, cfg)

            # 把 [T, N, ...] 展平成 [T*N, ...]，接下来像普通监督学习一样分 minibatch。
            b_obs = obs_buf.reshape((-1, obs_dim))
            b_actions = actions_buf.reshape(-1)
            b_logprobs = logprobs_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            explained_var = 1.0 - torch.var(b_returns - b_values) / (torch.var(b_returns) + 1e-8)
            # 标准化 advantage 不改变正负含义，但能让梯度尺度更稳。
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            policy_loss_value = 0.0
            value_loss_value = 0.0
            entropy_value = 0.0
            approx_kl_value = 0.0
            clipfrac_value = 0.0
            rnd_loss_value = 0.0
            # 第二阶段：固定这批 rollout 数据，对策略和价值函数做多轮小批量优化。
            for _ in range(cfg.epochs):
                indices = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    mb = indices[start:start + minibatch_size]
                    new_logprobs, entropy, new_values = model.evaluate(b_obs[mb], b_actions[mb])
                    # ratio = pi_new(a|s) / pi_old(a|s)，衡量新策略对旧动作概率改了多少。
                    logratio = new_logprobs - b_logprobs[mb]
                    ratio = logratio.exp()
                    # approx_kl/clipfrac 是诊断指标：太大通常说明策略更新过猛。
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                    # PPO clipped objective：advantage 为正时希望增大动作概率，为负时希望减小；
                    # clamp 会阻止 ratio 偏离 1 太远，从而避免一次 update 把策略改坏。
                    unclipped = -b_advantages[mb] * ratio
                    clipped = -b_advantages[mb] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    policy_loss = torch.max(unclipped, clipped).mean()
                    # critic 学习预测 return；乘 0.5 是 MSE 常见写法，方便求导时抵消平方的 2。
                    value_loss = 0.5 * ((new_values - b_returns[mb]) ** 2).mean()
                    # entropy 越大，策略越随机；总 loss 里减去 entropy，相当于奖励探索。
                    entropy_loss = entropy.mean()
                    loss = policy_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    policy_loss_value = float(policy_loss.detach().cpu())
                    value_loss_value = float(value_loss.detach().cpu())
                    entropy_value = float(entropy_loss.detach().cpu())
                    approx_kl_value = float(approx_kl.detach().cpu())
                    clipfrac_value = float(clipfrac.detach().cpu())

                    if rnd_model and rnd_optimizer:
                        prediction, target = rnd_model(b_obs[mb])
                        rnd_loss = 0.5 * ((prediction - target.detach()) ** 2).mean()
                        rnd_optimizer.zero_grad()
                        rnd_loss.backward()
                        nn.utils.clip_grad_norm_(rnd_model.predictor.parameters(), 0.5)
                        rnd_optimizer.step()
                        rnd_loss_value = float(rnd_loss.detach().cpu())

            update_elapsed = time.perf_counter() - update_started
            rnd_rewards_buf = intrinsic_rewards_buf * rnd_beta
            rnd_reward_abs_ratio = float(
                rnd_rewards_buf.abs().mean().cpu() / (extrinsic_rewards_buf.abs().mean().cpu() + 1e-8)
            )
            final_distance_mean = (
                float(np.mean(final_distance_samples))
                if final_distance_samples
                else None
            )
            episode_return_mean = (
                float(np.mean(episode_return_samples))
                if episode_return_samples
                else None
            )
            explained_var_value = float(explained_var.detach().cpu())

            metrics = {
                "charts/update_seconds": update_elapsed,
                "charts/episodes": float(episode_finishes),
                "losses/policy_loss": policy_loss_value,
                "losses/value_loss": value_loss_value,
                "losses/entropy": entropy_value,
                "losses/approx_kl": approx_kl_value,
                "losses/clipfrac": clipfrac_value,
                "losses/explained_variance": explained_var_value,
                "hyperparams/learning_rate": cfg.learning_rate,
            }
            if final_distance_mean is not None:
                metrics["charts/final_distance_mean"] = final_distance_mean
            if episode_return_mean is not None:
                metrics["charts/episode_return_mean"] = episode_return_mean
            logger.log(global_step, metrics)
            if rnd_model:
                logger.log(
                    global_step,
                    {
                        "charts/rnd_reward_abs_ratio": rnd_reward_abs_ratio,
                        "losses/rnd_loss": rnd_loss_value,
                        "hyperparams/rnd_beta": rnd_beta,
                    },
                )

            # 控制台输出只打印最新 update 的摘要；完整曲线看 TensorBoard。
            log_line = (
                f"update={update}/{cfg.updates} absolute_update={absolute_update} steps={global_step} "
                f"episodes={episode_finishes} "
                f"policy_loss={policy_loss_value:.4f} value_loss={value_loss_value:.4f} "
                f"entropy={entropy_value:.4f} kl={approx_kl_value:.5f} clipfrac={clipfrac_value:.3f}"
            )
            if final_distance_mean is not None:
                log_line += f" final_distance_mean={final_distance_mean:.3f}"
            if episode_return_mean is not None:
                log_line += f" episode_return_mean={episode_return_mean:.3f}"
            if rnd_model:
                log_line += (
                    f" rnd_reward_abs_ratio={rnd_reward_abs_ratio:.4f} "
                    f"rnd_beta={rnd_beta:.4f} rnd_loss={rnd_loss_value:.4f}"
                )
            print(log_line)

            if cfg.save_interval > 0 and update % cfg.save_interval == 0:
                # 周期性保存带 update 编号的 checkpoint，方便回看不同训练阶段的策略。
                periodic_path = cfg.checkpoint.with_name(f"{cfg.checkpoint.stem}_update_{absolute_update:06d}{cfg.checkpoint.suffix}")
                save_checkpoint(periodic_path, model, optimizer, cfg, absolute_update, global_step, rnd_model, rnd_optimizer)
                print(f"saved_checkpoint={periodic_path}")
    finally:
        logger.close()

    save_checkpoint(cfg.checkpoint, model, optimizer, cfg, resumed_update + cfg.updates, global_step, rnd_model, rnd_optimizer)
    print(f"saved_checkpoint={cfg.checkpoint}")
    print(f"logdir={log_dir}")
    print(f"tensorboard_logdir={log_dir}")


def main() -> int:
    # 命令行参数和 PPOConfig 字段一一对应；未传入的参数保持 None，让 make_config 使用默认/JSON 值。
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--minibatches", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-coef", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true", default=None)
    parser.add_argument(
        "--reward-profile",
        choices=["classic_100m", "hurdle_110m", "long_jump_30m"],
        default=None,
    )
    parser.add_argument("--game-mode", choices=["auto", "classic_100m", "hurdle_110m", "long_jump_30m"], default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=None)
    parser.add_argument("--frame-stack", type=int, default=None)
    parser.add_argument("--rnd-beta", type=float, default=None)
    parser.add_argument("--rnd-beta-final", type=float, default=None)
    parser.add_argument("--rnd-beta-decay-updates", type=int, default=None)
    parser.add_argument("--rnd-learning-rate", type=float, default=None)
    parser.add_argument("--rnd-hidden-size", type=int, default=None)
    parser.add_argument("--rnd-feature-dim", type=int, default=None)
    args = parser.parse_args()
    train(make_config(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
