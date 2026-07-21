from __future__ import annotations

from pathlib import Path
import platform
import sys

import numpy as np

_root_dir = Path(__file__).resolve().parents[1]
_bin_dir = _root_dir / "bin"
_extension_path = _bin_dir / "_running_env.cp310-win_amd64.pyd"
_physics_path = _bin_dir / "running_physics.dll"


def _validate_environment_files() -> None:
    if platform.system() != "Windows" or sys.maxsize <= 2**32:
        raise RuntimeError("The bundled training environment requires 64-bit Windows.")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The bundled training environment requires Python 3.10.")
    if not _physics_path.is_file():
        raise RuntimeError(f"Missing physics backend: {_physics_path.relative_to(_root_dir)}")
    if not _extension_path.is_file():
        raise RuntimeError(
            "Missing compiled training extension: bin/_running_env.cp310-win_amd64.pyd. "
            "Run: .\\.venv\\Scripts\\python.exe scripts\\build.py"
        )

_validate_environment_files()
if str(_bin_dir) not in sys.path:
    sys.path.insert(0, str(_bin_dir))

import _running_env


def infer_game_mode(reward_profile: str = "classic_100m") -> str:
    return str(_running_env.infer_game_mode(reward_profile))


class RunningGameEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_profile: str = "classic_100m",
        game_mode: str = "auto",
    ):
        self._env = _running_env.Env()
        self._env.configure(reward_profile, game_mode)
        self.reward_profile = reward_profile
        self.game_mode = self._env.game_mode
        self.observation_size = self._env.observation_size
        self.action_space_n = 16
        self.observation_shape = (self.observation_size,)
        try:
            import gymnasium as gym

            self.action_space = gym.spaces.Discrete(self.action_space_n)
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.observation_shape,
                dtype=np.float32,
            )
        except Exception:
            self.action_space = None
            self.observation_space = None

    def reset(self, *, seed=None, options=None):
        result = self._env.reset()
        return self._obs(result), self._info(result)

    def step(self, action):
        result = self._env.step(int(action))
        return (
            self._obs(result),
            result["reward"],
            result["done"],
            result["truncated"],
            self._info(result),
        )

    @staticmethod
    def _obs(result):
        return np.asarray(result["obs"], dtype=np.float32)

    @staticmethod
    def _info(result):
        return {
            "score": result["score"],
            "distance": result["distance"],
            "time": result["time"],
            "success": result["success"],
        }


class VectorRunningGameEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        num_threads: int = 1,
        reward_profile: str = "classic_100m",
        game_mode: str = "auto",
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        self._env = _running_env.BatchedEnv(num_envs, num_threads, reward_profile, game_mode)
        self.reward_profile = reward_profile
        self.game_mode = self._env.game_mode
        self.num_envs = num_envs
        self.num_threads = self._env.num_threads
        self.observation_size = self._env.observation_size
        self.single_observation_shape = (self.observation_size,)
        self.observation_shape = (num_envs, *self.single_observation_shape)
        self.action_space_n = 16
        try:
            import gymnasium as gym

            self.single_action_space = gym.spaces.Discrete(self.action_space_n)
            self.single_observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.single_observation_shape,
                dtype=np.float32,
            )
        except Exception:
            self.single_action_space = None
            self.single_observation_space = None

    def reset(self, *, seed=None, options=None):
        result = self._env.reset()
        obs = np.array(result["obs"], dtype=np.float32, copy=True)
        infos = self._infos(result)
        return obs, infos

    def step(self, actions):
        result = self.step_arrays(actions)
        obs = result["obs"]
        rewards = result["reward"]
        terminated = result["terminated"]
        truncated = result["truncated"]
        infos = self._infos(result)

        has_final = result["has_final"]
        reset_scores = result["reset_score"]
        reset_times = result["reset_time"]
        for i, done in enumerate(has_final):
            if not done:
                continue
            infos[i]["reset_info"] = {
                "score": float(reset_scores[i]),
                "time": float(reset_times[i]),
            }

        return obs, rewards, terminated, truncated, infos

    def step_arrays(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"actions shape must be ({self.num_envs},), got {actions.shape}")

        result = self._env.step(np.ascontiguousarray(actions))
        return {
            "obs": np.array(result["obs"], dtype=np.float32, copy=True),
            "reward": np.array(result["reward"], dtype=np.float32, copy=True),
            "terminated": np.array(result["done"], dtype=bool, copy=True),
            "truncated": np.array(result["truncated"], dtype=bool, copy=True),
            "score": np.array(result["score"], dtype=np.float32, copy=True),
            "time": np.array(result["time"], dtype=np.float32, copy=True),
            "has_final": np.array(result["has_final"], dtype=bool, copy=True),
            "final_score": np.array(result["final_score"], dtype=np.float32, copy=True),
            "final_distance": np.array(result["final_distance"], dtype=np.float32, copy=True),
            "final_time": np.array(result["final_time"], dtype=np.float32, copy=True),
            "final_success": np.array(result["final_success"], dtype=bool, copy=True),
            "reset_score": np.array(result["reset_score"], dtype=np.float32, copy=True),
            "reset_time": np.array(result["reset_time"], dtype=np.float32, copy=True),
        }

    def _infos(self, result):
        scores = result["score"] if isinstance(result, dict) else np.asarray(result["score"], dtype=np.float32)
        times = result["time"] if isinstance(result, dict) else np.asarray(result["time"], dtype=np.float32)
        return [
            {
                "score": float(scores[i]),
                "time": float(times[i]),
            }
            for i in range(self.num_envs)
        ]
