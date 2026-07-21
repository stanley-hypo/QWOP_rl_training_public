from __future__ import annotations

from running_env import RunningGameEnv


def main() -> int:
    print("environment=ok")
    for profile in ("classic_100m", "hurdle_110m", "long_jump_30m"):
        env = RunningGameEnv(reward_profile=profile)
        observation, _ = env.reset()
        observation, reward, terminated, truncated, info = env.step(0)
        print(
            f"profile={profile} obs={observation.shape[0]} reward={reward:.3f} "
            f"terminated={int(terminated)} truncated={int(truncated)} score={info['score']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
