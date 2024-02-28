import argparse
import numpy as np
from typing import Any, Tuple
from dataclasses import dataclass
from numpy.typing import NDArray
from peaceful_pie.unity_comms import UnityComms
from gym import Env, spaces
from stable_baselines3.common.env_checker import check_env


@dataclass
class MyVector3:
    x: float
    y: float
    z: float


@dataclass
class RLResult:
    reward: float
    finished: bool
    observation: MyVector3


class MyEnv(Env):
    def __init__(self, unity_comms: UnityComms):
        self.unity_comms = unity_comms
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def step(self, action: NDArray[np.uint8]) -> Tuple[NDArray[np.float32], float, bool, dict[str, Any]]:
        action_str = ["up", "down", "left", "right"][action]
        rl_result: RLResult = self.unity_comms.step(
            action=action_str, ResultClass=RLResult)
        info = {"finished": rl_result.finished}
        return self._obs_vec3_to_np_array(rl_result.observation), rl_result.reward, rl_result.finished, info

    def reset(self) -> NDArray[np.float32]:
        obs_vec3: MyVector3 = self.unity_comms.reset(ResultClass=MyVector3)
        return self._obs_vec3_to_np_array(obs_vec3)

    def _obs_vec3_to_np_array(self, vec3: MyVector3) -> NDArray[np.float32]:
        return np.array([vec3.x, vec3.y, vec3.z], dtype=np.float32)


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
