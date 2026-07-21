from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]

from running_env import VectorRunningGameEnv, infer_game_mode
from train_ppo import ActorCritic, FrameStacker, PPOConfig, load_model_compatible, parse_hidden_sizes


def checkpoint_config(data: dict) -> dict:
    config = data.get("config", {})
    return config if isinstance(config, dict) else {}


def hidden_sizes_from_config(config: dict) -> tuple[int, ...]:
    raw = config.get("hidden_sizes", (128, 128))
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("(") and raw.endswith(")"):
            raw = raw[1:-1]
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(value) for value in raw)


def frame_stack_from_config(config: dict) -> int:
    return max(1, int(config.get("frame_stack", 1)))


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def collect_teacher_batch(
    env: VectorRunningGameEnv,
    stacker: FrameStacker,
    teacher: ActorCritic,
    obs: torch.Tensor,
    device: torch.device,
    batch_steps: int,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_samples: list[torch.Tensor] = []
    teacher_logits: list[torch.Tensor] = []
    for _ in range(batch_steps):
        obs_samples.append(obs.detach().cpu())
        with torch.no_grad():
            logits, _ = teacher(obs)
            teacher_logits.append(logits.detach().cpu())
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = torch.distributions.Categorical(logits=logits).sample()
        step_result = env.step_arrays(actions.cpu().numpy())
        done_np = np.logical_or(step_result["terminated"], step_result["truncated"])
        next_obs_np = stacker.update(step_result["obs"], done_np)
        obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    return torch.cat(obs_samples, dim=0).to(device), torch.cat(teacher_logits, dim=0).to(device), obs


def save_student_checkpoint(
    path: Path,
    teacher_path: Path,
    student: ActorCritic,
    optimizer: torch.optim.Optimizer,
    teacher_data: dict,
    student_hidden_sizes: tuple[int, ...],
    frame_stack: int,
    reward_profile: str,
    game_mode: str,
) -> None:
    teacher_config = checkpoint_config(teacher_data)
    valid_config_fields = {field.name for field in fields(PPOConfig)}
    config = {key: value for key, value in teacher_config.items() if key in valid_config_fields}
    config["hidden_sizes"] = list(student_hidden_sizes)
    config["frame_stack"] = frame_stack
    config["reward_profile"] = reward_profile
    config["game_mode"] = game_mode
    config["distilled_from"] = str(teacher_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "update": int(teacher_data.get("update", 0)),
            "global_step": int(teacher_data.get("global_step", 0)),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-resume", type=Path, default=None)
    parser.add_argument("--student-hidden-sizes", type=parse_hidden_sizes, default=(256, 256))
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--batch-steps", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    teacher_path = resolve_path(args.teacher)
    output_path = resolve_path(args.output)
    student_resume_path = resolve_path(args.student_resume) if args.student_resume else None
    teacher_data = torch.load(teacher_path, map_location=device)
    teacher_config = checkpoint_config(teacher_data)
    reward_profile = str(teacher_config.get("reward_profile", "classic_100m"))
    game_mode = str(teacher_config.get("game_mode", "auto"))
    if game_mode == "auto":
        game_mode = infer_game_mode(reward_profile)
    frame_stack = frame_stack_from_config(teacher_config)

    env = VectorRunningGameEnv(
        args.num_envs,
        num_threads=args.num_threads,
        reward_profile=reward_profile,
        game_mode=game_mode,
    )
    base_obs_np, _ = env.reset()
    stacker = FrameStacker(args.num_envs, base_obs_np.shape[1], frame_stack)
    obs_np = stacker.reset(base_obs_np)
    obs_dim = stacker.obs_dim
    action_dim = env.action_space_n

    teacher = ActorCritic(obs_dim, action_dim, hidden_sizes_from_config(teacher_config)).to(device)
    teacher.load_state_dict(teacher_data["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = ActorCritic(obs_dim, action_dim, args.student_hidden_sizes).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate, eps=1e-5)
    if student_resume_path:
        student_data = torch.load(student_resume_path, map_location=device)
        student.load_state_dict(student_data["model"])
        try:
            optimizer.load_state_dict(student_data["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        except (KeyError, ValueError, RuntimeError):
            print("student_optimizer_state_skipped=1 reason=incompatible_or_missing")
        print(f"student_resumed_from={student_resume_path}")
    else:
        load_model_compatible(student, teacher_data["model"])

    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    for iteration in range(1, args.iterations + 1):
        b_obs, b_teacher_logits, obs = collect_teacher_batch(
            env,
            stacker,
            teacher,
            obs,
            device,
            args.batch_steps,
            args.deterministic,
        )
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(b_teacher_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            _, teacher_values = teacher(b_obs)

        indices = torch.randperm(b_obs.shape[0], device=device)
        policy_loss_value = 0.0
        value_loss_value = 0.0
        for start in range(0, b_obs.shape[0], args.minibatch_size):
            mb = indices[start : start + args.minibatch_size]
            student_logits, student_values = student(b_obs[mb])
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            policy_loss = F.kl_div(student_log_probs, teacher_probs[mb], reduction="batchmean")
            value_loss = 0.5 * ((student_values - teacher_values[mb]) ** 2).mean()
            loss = policy_loss + args.value_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            policy_loss_value = float(policy_loss.detach().cpu())
            value_loss_value = float(value_loss.detach().cpu())

        print(
            f"iteration={iteration}/{args.iterations} samples={b_obs.shape[0]} "
            f"policy_kl={policy_loss_value:.6f} value_loss={value_loss_value:.6f}"
        )

    save_student_checkpoint(
        output_path,
        teacher_path,
        student,
        optimizer,
        teacher_data,
        args.student_hidden_sizes,
        frame_stack,
        reward_profile,
        env.game_mode,
    )
    print(f"saved_student={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
