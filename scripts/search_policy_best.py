from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from export_policy_replay import (
    checkpoint_config,
    frame_stack_from_config,
    hidden_sizes_from_config,
    is_better,
    resolve_path,
)
from replay_io import ruleset_for_game_mode, write_replay
from running_env import VectorRunningGameEnv, infer_game_mode
from train_ppo import ActorCritic


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample checkpoint rollouts and save the best binary replay.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=128)
    args = parser.parse_args()
    if args.episodes <= 0 or args.num_envs <= 0 or args.num_threads <= 0:
        raise ValueError("episodes, num_envs, and num_threads must be positive")

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

    env = VectorRunningGameEnv(
        args.num_envs,
        num_threads=args.num_threads,
        reward_profile=reward_profile,
        game_mode=game_mode,
    )
    obs, _ = env.reset()
    obs_dim = obs.shape[1] * frame_stack
    model = ActorCritic(obs_dim, env.action_space_n, hidden_sizes_from_config(config)).to(device)
    model.load_state_dict(data["model"])
    model.eval()

    frames = np.repeat(obs[:, None, :], frame_stack, axis=1)
    histories: list[list[int]] = [[] for _ in range(env.num_envs)]
    best = None
    finished = 0
    successes = 0
    last_reported = 0
    while finished < args.episodes:
        stacked_obs = frames.reshape(env.num_envs, obs_dim)
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(stacked_obs, dtype=torch.float32, device=device))
            if args.deterministic:
                actions_tensor = torch.argmax(logits, dim=-1)
            else:
                actions_tensor = torch.distributions.Categorical(logits=logits).sample()
        actions = actions_tensor.cpu().numpy().astype(np.int64)
        for env_index, action in enumerate(actions.tolist()):
            histories[env_index].append(int(action))

        result = env.step_arrays(actions)
        obs = result["obs"]
        frames = np.roll(frames, shift=-1, axis=1)
        frames[:, -1, :] = obs
        for env_index, ended in enumerate(result["has_final"].tolist()):
            if not ended:
                continue
            if finished >= args.episodes:
                break
            finished += 1
            candidate = {
                "actions": histories[env_index].copy(),
                "score": float(result["final_score"][env_index]),
                "distance": float(result["final_distance"][env_index]),
                "time": float(result["final_time"][env_index]),
                "success": bool(result["final_success"][env_index]),
                "truncated": bool(result["truncated"][env_index]),
            }
            successes += int(candidate["success"])
            if is_better(candidate, best, env.game_mode):
                best = candidate
                print(
                    f"episode={finished} new_best steps={len(candidate['actions'])} "
                    f"score={candidate['score']:.3f} distance={candidate['distance']:.3f} "
                    f"time={candidate['time']:.3f} success={int(candidate['success'])}",
                    flush=True,
                )
            histories[env_index] = []
            frames[env_index, :, :] = obs[env_index]
        if args.progress_interval > 0 and finished >= last_reported + args.progress_interval:
            print(f"progress={finished}/{args.episodes} successes={successes}", flush=True)
            last_reported = finished

    if best is None:
        raise RuntimeError("no episodes were evaluated")
    write_replay(output, best["actions"], ruleset_for_game_mode(env.game_mode))
    print(
        f"saved_replay={output} episodes={finished} successes={successes} "
        f"steps={len(best['actions'])} score={best['score']:.3f} "
        f"distance={best['distance']:.3f} time={best['time']:.3f} "
        f"success={int(best['success'])} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
