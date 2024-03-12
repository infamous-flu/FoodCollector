import argparse
import numpy as np
from typing import Any, Tuple, List
from dataclasses import dataclass
from numpy.typing import NDArray
from peaceful_pie.unity_comms import UnityComms
from gym import Env, spaces
from stable_baselines3.common.env_checker import check_env


@dataclass
class RayResults:
    rayDistances: List[List[float]]
    rayHitObjectTypes: List[List[int]]
    NumObjectTypes: int


@dataclass
class MyVector3:
    x: float
    y: float
    z: float


@dataclass
class RLResult:
    reward: float
    finished: bool
    observation: RayResults


class MyEnv(Env):
    def __init__(self, unity_comms: UnityComms):
        self.unity_comms = unity_comms
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(60, 3), dtype=np.float32)

    def step(self, action: NDArray[np.uint8]) -> Tuple[NDArray[np.float32], float, bool, dict[str, Any]]:
        action_str = ["forward", "left", "right"][action]
        rl_result: RLResult = self.unity_comms.step(
            action=action_str, ResultClass=RLResult)
        info = {"finished": rl_result.finished}
        # print(self._ray_results_to_np_array(rl_result.observation))
        return self._ray_results_to_np_array(rl_result.observation), rl_result.reward, rl_result.finished, info

    def reset(self) -> NDArray[np.float32]:
        rl_result: RLResult = self.unity_comms.reset(ResultClass=RLResult)
        # print(self._ray_results_to_np_array(rl_result.observation))
        return self._ray_results_to_np_array(rl_result.observation)

    def _ray_results_to_np_array(self, ray_results: RayResults):
        distances_np = np.array(ray_results.rayDistances)
        distances_np = 1 / distances_np
        object_types_np = np.array(ray_results.rayHitObjectTypes)

        _obs = np.zeros(
            (ray_results.NumObjectTypes + 1, *distances_np.shape), dtype=np.float32
        )
        np.put_along_axis(
            _obs,
            np.expand_dims(object_types_np, axis=0),
            np.expand_dims(distances_np, axis=0),
            axis=0,
        )

        _obs = _obs[:-1]
        return _obs


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port=args.port)
    my_env = MyEnv(unity_comms=unity_comms)
    check_env(env=my_env)
    my_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
