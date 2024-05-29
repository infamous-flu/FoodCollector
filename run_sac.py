import torch
from mlagents_envs.environment import UnityEnvironment

from envs.my_env import MyEnv
from agents.sac import SAC, SACConfig
from utils.string_log_channel import StringLogChannel

SEED = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AGENT_CONFIG = SACConfig()

N_EPISODES = 10
MAX_EPISODES_PER_EPISODE = 1000

CENTRING = 80


def rollout(env, sac):
    episode_return = 0
    obs, _ = env.reset()
    h, c = sac.rnn.initial_state(1)
    with torch.no_grad():
        latent = sac.vae.encode(torch.from_numpy(obs).unsqueeze(0)).squeeze(0)
    state = torch.cat([latent, h.squeeze()], dim=-1)
    for _ in range(MAX_EPISODES_PER_EPISODE):
        action, _, _ = sac.actor.get_action(state.unsqueeze(0).to(DEVICE))
        action = action.detach().cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        with torch.no_grad():
            latent = sac.vae.encode(torch.from_numpy(next_obs).unsqueeze(0)).squeeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            _, _, _, h, c = sac.rnn(latent.unsqueeze(0).unsqueeze(1), action_tensor, h, c)
        next_state = torch.cat([latent, h.squeeze()], dim=-1)
        episode_return += reward
        done = terminated or truncated
        if done:
            break
        state = next_state
    return episode_return


def main():
    try:
        uuid_str = 'a1d8f7b7-cec8-50f9-b78b-d3e165a78520'
        channel = StringLogChannel(uuid_str)
        unity_env = UnityEnvironment(file_name=None, seed=SEED, side_channels=[channel])
        unity_env.reset()
        print('\nUnity Environment connected')
        gym_env = MyEnv(unity_env)
        sac = SAC(gym_env, DEVICE, AGENT_CONFIG, SEED)
        sac.load_model('saved_models/sac/model_2024-05-29-02h35m.pth')
        print('\n    ' + 'Episode Return'.center(CENTRING))
        div = '─' * ((CENTRING * 2) // 3)
        print('    ' + div.center(CENTRING))
        for eps in range(N_EPISODES):
            episode_return = rollout(gym_env, sac)
            res = f'Episode {eps + 1:>2}: {episode_return:>5}'
            print('    ' + res.center(CENTRING))
    except Exception as e:
        print(f'\nAn error occurred: {e}')
    finally:
        unity_env.close()
        print('\nEnvironment closed')


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
