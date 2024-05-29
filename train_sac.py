import torch
from mlagents_envs.environment import UnityEnvironment

from envs.my_env import MyEnv
from agents.sac import SAC, SACConfig, TrainingConfig
from utils.string_log_channel import StringLogChannel


SEED = 69420
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AGENT_CONFIG = SACConfig()
TRAINING_CONFIG = TrainingConfig()


def main():
    try:
        uuid_str = 'a1d8f7b7-cec8-50f9-b78b-d3e165a78520'
        channel = StringLogChannel(uuid_str)
        unity_env = UnityEnvironment(file_name=None, seed=SEED, side_channels=[channel])
        unity_env.reset()
        print('\nUnity Environment connected')
        gym_env = MyEnv(unity_env)
        sac = SAC(gym_env, DEVICE, AGENT_CONFIG, SEED)
        sac.learn(TRAINING_CONFIG)
    except Exception as e:
        print(f'\nAn error occurred: {e}')
    finally:
        sac.save_model(TRAINING_CONFIG.save_path)
        unity_env.close()
        print('\nEnvironment closed')


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
