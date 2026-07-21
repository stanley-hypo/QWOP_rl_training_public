from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from replay_io import ruleset_for_game_mode, write_replay
from running_env import RunningGameEnv, infer_game_mode
from train_ppo import ActorCritic


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def checkpoint_config(data: dict) -> dict:
    config = data.get("config", {})
    return config if isinstance(config, dict) else {}


def hidden_sizes_from_config(config: dict) -> tuple[int, ...]:
    raw = config.get("hidden_sizes", (128, 128))
    if isinstance(raw, str):
        raw = raw.strip().strip("[]()")
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(int(value) for value in raw)


def frame_stack_from_config(config: dict) -> int:
    return max(1, int(config.get("frame_stack", 1)))


def choose_action(model: ActorCritic, obs: np.ndarray, device: torch.device, stochastic: bool) -> int:
    obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _ = model(obs_tensor)
        if not stochastic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())


def rollout(
    env: RunningGameEnv,
    model: ActorCritic,
    device: torch.device,
    max_steps: int,
    stochastic: bool,
    frame_stack: int,
) -> dict:
    obs, info = env.reset()
    frames = [obs.copy() for _ in range(frame_stack)]
    actions: list[int] = []
    done = False
    truncated = False
    for _ in range(max_steps):
        action = choose_action(model, np.concatenate(frames), device, stochastic)
        obs, _, done, truncated, info = env.step(action)
        frames.pop(0)
        frames.append(obs.copy())
        actions.append(action)
        if done or truncated:
            break
    return {
        "actions": actions,
        "score": float(info["score"]),
        "distance": float(info["distance"]),
        "time": float(info["time"]),
        "success": bool(info["success"]),
        "truncated": bool(truncated),
    }


def is_better(candidate: dict, incumbent: dict | None, game_mode: str) -> bool:
    if incumbent is None:
        return True
    if candidate["success"] != incumbent["success"]:
        return candidate["success"]
    if game_mode == "long_jump_30m":
        if candidate["score"] != incumbent["score"]:
            return candidate["score"] > incumbent["score"]
    elif candidate["success"] and candidate["time"] != incumbent["time"]:
        return candidate["time"] < incumbent["time"]
    elif candidate["distance"] != incumbent["distance"]:
        return candidate["distance"] > incumbent["distance"]
    return candidate["time"] < incumbent["time"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export checkpoint policy actions to a binary replay.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("replays/policy_replay.bin"))
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.episodes <= 0:
        raise ValueError("max_steps and episodes must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    ))
    checkpoint = resolve_path(args.checkpoint)
    output = resolve_path(args.output)
    data = torch.load(checkpoint, map_location=device)
    config = checkpoint_config(data)
    reward_profile = str(config.get("reward_profile", "classic_100m"))
    game_mode = str(config.get("game_mode", "auto"))
    if game_mode == "auto":
        game_mode = infer_game_mode(reward_profile)
    frame_stack = frame_stack_from_config(config)

    env = RunningGameEnv(reward_profile=reward_profile, game_mode=game_mode)
    obs, _ = env.reset()
    model = ActorCritic(
        obs.shape[0] * frame_stack,
        env.action_space_n,
        hidden_sizes_from_config(config),
    ).to(device)
    model.load_state_dict(data["model"])
    model.eval()

    best = None
    for episode in range(1, args.episodes + 1):
        candidate = rollout(env, model, device, args.max_steps, args.stochastic, frame_stack)
        if is_better(candidate, best, env.game_mode):
            best = candidate
        print(
            f"episode={episode} steps={len(candidate['actions'])} score={candidate['score']:.3f} "
            f"distance={candidate['distance']:.3f} time={candidate['time']:.3f} "
            f"success={int(candidate['success'])}"
        )

    if best is None:
        raise RuntimeError("no episodes were evaluated")
    write_replay(output, best["actions"], ruleset_for_game_mode(env.game_mode))
    print(
        f"saved_replay={output} steps={len(best['actions'])} score={best['score']:.3f} "
        f"distance={best['distance']:.3f} time={best['time']:.3f} "
        f"success={int(best['success'])} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
