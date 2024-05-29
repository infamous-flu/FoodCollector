import numpy as np
import torch
import gymnasium as gym
from typing import Tuple
from numpy.typing import NDArray
from mlagents_envs.environment import BaseEnv, UnityEnvironment, ActionTuple
from stable_baselines3.common.env_checker import check_env

from utils.string_log_channel import StringLogChannel


class MyEnv(gym.Env):
    def __init__(self, unity_env: BaseEnv):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.unity_env = unity_env
        self.behavior_name = list(self.unity_env.behavior_specs)[0]
        self.behavior_spec = self.unity_env.behavior_specs[self.behavior_name]

    def step(self, action: NDArray[np.float32]) -> Tuple[NDArray[np.float32], float, bool, bool, dict]:
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action.reshape(1, 2))
        self.unity_env.set_actions(self.behavior_name, action_tuple)
        self.unity_env.step()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        if self.tracked_agent in terminal_steps:
            obs = terminal_steps[self.tracked_agent].obs[0]
            reward = terminal_steps[self.tracked_agent].reward
            truncated = terminal_steps[self.tracked_agent].interrupted
            terminated = not truncated
        if self.tracked_agent in decision_steps:
            obs = decision_steps[self.tracked_agent].obs[0]
            reward = decision_steps[self.tracked_agent].reward
            terminated, truncated = False, False
        return obs, float(reward), terminated, truncated, dict()

    def reset(self, seed=None) -> Tuple[NDArray[np.float32], dict]:
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        self.tracked_agent = decision_steps.agent_id[0]
        obs = decision_steps[self.tracked_agent].obs[0]
        return obs, dict()


if __name__ == '__main__':
    try:
        uuid_str = 'a1d8f7b7-cec8-50f9-b78b-d3e165a78520'
        channel = StringLogChannel(uuid_str)
        unity_env = UnityEnvironment(file_name=None, side_channels=[channel])
        unity_env.reset()
        gym_unity_env = MyEnv(unity_env)
        check_env(gym_unity_env)
    except Exception as e:
        print(f'An error occurred: {e}')
    finally:
        unity_env.close()
