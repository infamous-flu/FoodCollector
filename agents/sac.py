import os
import random
from datetime import datetime
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from utils.helper import nice_box
from networks.vae import ConvVAE
from networks.mdnrnn import MDNRNN


@dataclass
class SACConfig:
    """
    Configuration class for Soft Actor-Critic (SAC) hyperparameters.

    Attributes:
        buffer_size (int): Maximum size of the replay buffer.
        gamma (float): Discount factor for future rewards.
        tau (float): Target smoothing coefficient for soft updates.
        batch_size (int): Number of samples per batch for training.
        learning_starts (int): Number of steps before training starts.
        policy_lr (float): Learning rate for the policy network.
        q_lr (float): Learning rate for the Q-networks.
        policy_freq (int): Frequency of policy updates.
        target_freq (int): Frequency of target network.
        vae_path (str): Path to the pre-trained VAE model.
        rnn_path (str): Path to the pre-trained RNN model.
    """

    buffer_size: int = int(1e5)
    gamma: float = 0.99
    tau: float = 5e-3
    batch_size: int = 100
    learning_starts: int = 10000
    policy_lr: float = 5e-4
    q_lr: float = 1e-3
    policy_freq: int = 2
    target_freq: int = 1
    vae_path: str = os.path.join('saved_models', 'vae', 'vae_model_32.pth')
    rnn_path: str = os.path.join('saved_models', 'rnn', 'rnn_model_128.pth')


@dataclass
class TrainingConfig:
    """
    Configuration class for SAC training parameters.

    Attributes:
        env_id (str): Identifier for the environment to train in.
        n_timesteps (int): Total number of training timesteps.
        seed (Optional[int]): Random seed for reproducibility.
        max_timesteps_per_episode (int): Maximum timesteps per episode.
        window_size (int): Window size for tracking average returns.
        print_freq (int): Frequency of printing training updates.
        enable_logging (bool): Flag to enable TensorBoard logging.
        log_dir (str): Directory for saving TensorBoard logs.
        checkpoint_freq (int): Frequency of saving model checkpoints.
        save_path (str): Path for saving the final model.
    """

    env_id: str = 'FoodCollector'
    n_timesteps: int = int(1e6)
    seed: Optional[int] = None
    max_timesteps_per_episode: int = 1000
    window_size: int = 10
    print_freq: int = 10000
    enable_logging: bool = True
    log_dir: str = os.path.join('runs', 'sac', datetime.now().strftime('%Y-%m-%d-%Hh%Mm'))
    checkpoint_freq: int = 10000
    save_path: str = os.path.join('saved_models', 'sac', f'model_{datetime.now().strftime("%Y-%m-%d-%Hh%Mm")}.pth')


class ReplayBuffer:
    """
    Replay buffer for storing and sampling agent experiences.

    Attributes:
        buffer_size (int): Maximum size of the replay buffer.
        state_memory (np.ndarray): Array to store state observations.
        action_memory (np.ndarray): Array to store actions taken.
        next_state_memory (np.ndarray): Array to store next state observations.
        reward_memory (np.ndarray): Array to store rewards received.
        done_memory (np.ndarray): Array to store episode termination flags.
        pointer (int): Current position to insert the next experience.
        size (int): Current number of experiences stored in the buffer.
    """

    def __init__(self, buffer_size: int, n_observations: int, n_actions: int):
        """
        Initialize the replay buffer.

        Args:
            buffer_size (int): Maximum size of the replay buffer.
            n_observations (int): Number of observation dimensions.
            n_actions (int): Number of action dimensions.
        """

        self.buffer_size = buffer_size
        self.state_memory = np.zeros((buffer_size, n_observations), dtype=np.float32)
        self.action_memory = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, n_observations), dtype=np.float32)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.done_memory = np.zeros(buffer_size, dtype=np.bool)
        self.pointer = 0
        self.size = 0

    def add(self, state: NDArray[np.float32], action: NDArray[np.float32],
            next_state: NDArray[np.float32], reward: float, done: bool):
        """
        Add a new experience to the replay buffer.

        Args:
            state (NDArray[np.float32]): Current state observation.
            action (NDArray[np.float32]): Action taken.
            next_state (NDArray[np.float32]): Next state observation.
            reward (float): Reward received.
            done (bool): Episode termination flag.
        """

        index = self.pointer % self.buffer_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.pointer += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32],
                                               NDArray[np.float32], NDArray[np.bool]]:
        """
        Sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[bool]]:
            Batch of sampled experiences (states, actions, next_states, rewards, dones).
        """

        indices = np.random.choice(self.size, batch_size, replace=False)
        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        next_states = self.next_state_memory[indices]
        rewards = self.reward_memory[indices]
        dones = self.done_memory[indices]

        return states, actions, next_states, rewards, dones

    def __len__(self) -> int:
        """
        Get the current size of the replay buffer.

        Returns:
            int: Current number of experiences stored.
        """

        return self.size


class SoftQNetwork(nn.Module):
    """
    Soft Q-Network for the SAC agent.

    This network approximates the Q-value function, taking both the state and action as inputs.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Output layer producing Q-value.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initialize the Soft Q-Network.

        Args:
            n_observations (int): Number of observation dimensions.
            n_actions (int): Number of action dimensions.
        """

        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(n_observations + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.tensor, a: torch.tensor) -> torch.tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.tensor): State input tensor.
            a (torch.tensor): Action input tensor.

        Returns:
            torch.tensor: Q-value for the given state-action pair.
        """

        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class Actor(nn.Module):
    """
    Policy network for the SAC agent.

    This network generates actions given the state input, using a stochastic policy.

    Attributes:
        LOG_STD_MAX (float): Maximum log standard deviation.
        LOG_STD_MIN (float): Minimum log standard deviation.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc_mean (nn.Linear): Fully connected layer to output the mean of the action distribution.
        fc_log_std (nn.Linear): Fully connected layer to output the log standard deviation of the \
            action distribution.
        action_scale (torch.tensor): Scale of the actions.
        action_bias (torch.tensor): Bias of the actions.
    """

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, n_observations: int, n_actions: int, action_space):
        """
        Initialize the policy network.

        Args:
            n_observations (int): Number of observation dimensions.
            n_actions (int): Number of action dimensions.
            action_space: Action space object containing action range information.
        """

        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_observations, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, n_actions)
        self.fc_log_std = nn.Linear(256, n_actions)
        self.register_buffer(
            'action_scale', torch.tensor(action_space.high - action_space.low / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            'action_bias', torch.tensor(action_space.high + action_space.low / 2.0, dtype=torch.float32)
        )

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Forward pass through the network.

        Args:
            x (torch.tensor): State input tensor.

        Returns:
            Tuple[torch.tensor, torch.tensor]: Mean and log standard deviation of the action distribution.
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_log_std(x))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Sample an action from the policy given the state.

        Args:
            state (torch.tensor): State input tensor.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]: Sampled action, log probability of the action, \
                and mean action.
        """

        mean, log_std = self(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-10)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


class SAC:
    """
    Soft Actor-Critic (SAC) agent.

    This class implements the SAC algorithm, which includes initializing the environment,
    networks, hyperparameters, and handling the training process.
    """

    def __init__(self, env: gym.Env, device: torch.device, agent_config: SACConfig, seed: Optional[int] = None):
        """
        Initialize the SAC agent.

        Args:
            env (gym.Env): The environment the agent will interact with.
            device (torch.device): The device (CPU or GPU) to run the computations on.
            agent_config (SACConfig): Configuration for SAC hyperparameters.
            seed (Optional[int]): Random seed for reproducibility.
        """

        self.env = env
        self.device = device
        self.agent_config = agent_config
        if seed is not None:
            self._set_seed(seed)
        self._init_vae(agent_config.vae_path)
        self._init_rnn(agent_config.rnn_path)
        self.n_observations = self.rnn.latent_dim + self.rnn.hidden_dim
        self.n_actions = env.action_space.shape[0]
        self._init_hyperparameters()
        self._init_networks()
        self._init_alpha()
        self.memory = ReplayBuffer(self.buffer_size, self.n_observations, self.n_actions)

    def learn(self, training_config: TrainingConfig):
        """
        Train the SAC agent.

        Args:
            training_config (TrainingConfig): Configuration for training parameters.
        """

        self.training_config = training_config         # Configuration for training parameters
        self._init_training_parameters()
        if self.training_config.seed is not None:      # Initialize training parameters
            self._set_seed(self.training_config.seed)  # Set the seed in various components
        if self.enable_logging:
            self.writer = SummaryWriter(self.log_dir)  # Prepare the TensorBoard writer for logging

        self.t = 0
        self.returns_window = deque([], maxlen=self.window_size)

        box_width = 82
        print('\n' + nice_box(
            width=box_width,
            contents=[(f'SAC', 'c')]
            + [('─'*(box_width-6), 'c')]
            + [('  Training settings:', '')]
            + [(f'    - {key}: {value}', '') for key, value in vars(self.training_config).items()]
            + [('─'*(box_width-6), 'c')]
            + [('  Agent hyperparameters:', '')]
            + [(f'    - {key}: {value}', '') for key, value in vars(self.agent_config).items()],
            padding=4, thick=True) + '\n'
        )
        print('    ' + 'Average Training Returns'.center(box_width) + '\n  ' + '─' * 88)

        while self.t < self.n_timesteps:
            self.rollout()  # Perform one episode of interaction with the environment

        # Final save and close the logger
        if self.checkpoint_freq > 0:
            self.save_model(self.save_path)
        if self.enable_logging:
            self.writer.close()

        self.env.close()

    def rollout(self):
        """
        Perform one episode of interaction with the environment.

        This method runs a single episode, interacting with the environment, collecting experiences,
        storing them in the replay buffer, and training the agent periodically.
        """

        episode_return = 0
        observation, _ = self.env.reset()  # Reset the environment and get the initial observation
        h, c = self.rnn.initial_state(1)   # Initialize the RNN hidden and cell states

        # Encode the initial observation using the VAE
        with torch.no_grad():
            latent = self.vae.encode(torch.from_numpy(observation).unsqueeze(0)).squeeze(0)
        state = torch.cat([latent, h.squeeze()], dim=-1)  # Concatenate the latent obs and hidden state

        for _ in range(self.max_timesteps_per_episode):
            self.t += 1

            # Select an action
            if self.t < self.learning_starts:
                action = self.env.action_space.sample()  # Sample random action if still in the initial phase
            else:
                action, _, _ = self.actor.get_action(state.unsqueeze(0).to(self.device))  # Get action from policy
                action = action.detach().cpu().numpy()

            # Step the environment
            next_observation, reward, terminated, truncated, _ = self.env.step(action)

            # Encode the next observation using the VAE
            with torch.no_grad():
                latent = self.vae.encode(torch.from_numpy(next_observation).unsqueeze(0)).squeeze(0)
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                if action_tensor.dim() < 3:
                    action_tensor = action_tensor.unsqueeze(0)
                _, _, _, h, c = self.rnn(latent.unsqueeze(0).unsqueeze(1), action_tensor, h, c)
            next_state = torch.cat([latent, h.squeeze()], dim=-1)  # Concatenate the latent obs and hidden state

            episode_return += reward
            done = terminated or truncated

            # Add experience to replay buffer
            self.memory.add(state, action, next_state, reward, done)

            # Train the agent if enough experiences are collected
            if self.t >= self.learning_starts:
                self.train()

            # Print progress periodically
            if self.print_freq > 0 and self.t % self.print_freq == 0:
                str1 = f'Timestep {self.t:>8}'
                str2 = f'Average Training Return: {np.mean(self.returns_window):.3f}'
                res = str1.ljust(16) + '        ' + str2.ljust(32)
                print('    ' + res.center(82))

            if done:
                break

            state = next_state

        self.returns_window.append(episode_return)

        # Log episodic return if logging is enabled
        if self.enable_logging:
            self.writer.add_scalar('Episodic/EpisodeReturn', episode_return, self.t)

    def train(self):
        """
        Train the SAC agent by updating the Q-networks, policy, and entropy coefficient.

        This method samples a batch of experiences from the replay buffer, computes the target Q-values,
        updates the Q-networks using the MSE loss, periodically updates the policy and entropy coefficient,
        and performs a soft update of the target Q-networks.
        """

        # Sample a batch of experiences from the replay buffer
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute the target Q-values
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_states)
            qf1_next_target = self.qf1_target(next_states, next_state_actions)
            qf2_next_target = self.qf2_target(next_states, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        # Compute the current Q-values
        qf1_a_values = self.qf1(states, actions).view(-1)
        qf2_a_values = self.qf2(states, actions).view(-1)

        # Compute the Q-function losses
        qf1_loss = self.MSELoss(qf1_a_values, next_q_value)
        qf2_loss = self.MSELoss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Update the Q-networks
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # Periodically update the policy and entropy coefficient
        if self.t % self.policy_freq == 0:
            for _ in range(self.policy_freq):
                # Compute the policy loss
                pi, log_pi, _ = self.actor.get_action(states)
                qf1_pi = self.qf1(states, pi)
                qf2_pi = self.qf2(states, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                # Update the policy network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Compute the entropy coefficient loss
                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(states)
                alpha_loss = (-torch.exp(self.log_alpha) * (log_pi + self.target_entropy)).mean()

                # Update the entropy coefficient
                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()

                # Update the alpha value
                self.alpha = torch.exp(self.log_alpha).item()

            # Log training metrics if logging is enabled
            if self.enable_logging:
                self.writer.add_scalar('SoftQNetwork/QF1Values', qf1_a_values.mean().item(), self.t)
                self.writer.add_scalar('SoftQNetwork/QF2Values', qf2_a_values.mean().item(), self.t)
                self.writer.add_scalar('SoftQNetwork/QF1Loss', qf1_loss.item(), self.t)
                self.writer.add_scalar('SoftQNetwork/QF2Loss', qf1_loss.item(), self.t)
                self.writer.add_scalar('Actor/ActorLoss', actor_loss.item(), self.t)
                self.writer.add_scalar('Actor/Alpha', self.alpha, self.t)
                self.writer.add_scalar('Actor/AlphaLoss', alpha_loss.item(), self.t)

        # Periodically update the target Q-networks
        if self.t % self.target_freq == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Save the model periodically
        if self.checkpoint_freq > 0 and self.t % self.checkpoint_freq == 0:
            self.save_model(self.save_path)

    def save_model(self, save_path: str):
        """
        Save the current state of the model to a file.

        This method saves the state dictionaries of the actor, Q-networks, target Q-networks,
        and their respective optimizers to a file at the specified path.

        Args:
            save_path (str): Path to the file where the model state will be saved.
        """

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'qf2_target_state_dict': self.qf2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'a_optimizer_state_dict': self.a_optimizer.state_dict(),
        }, save_path)

    def load_model(self, save_path: str):
        """
        Load the model state from a file.

        This method loads the state dictionaries of the actor, Q-networks, target Q-networks,
        and their respective optimizers from a file at the specified path. It also restores
        the entropy coefficient (alpha).

        Args:
            save_path (str): Path to the file from which the model state will be loaded.
        """

        checkpoint = torch.load(save_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
        self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
        self.qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
        self.qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = torch.exp(self.log_alpha).item()
        self.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])

    def _init_hyperparameters(self):
        """
        Initialize SAC hyperparameters from the configuration.
        """

        self.buffer_size = self.agent_config.buffer_size
        self.gamma = self.agent_config.gamma
        self.tau = self.agent_config.tau
        self.batch_size = self.agent_config.batch_size
        self.learning_starts = self.agent_config.learning_starts
        self.policy_lr = self.agent_config.policy_lr
        self.q_lr = self.agent_config.q_lr
        self.policy_freq = self.agent_config.policy_freq
        self.target_freq = self.agent_config.target_freq

    def _init_networks(self):
        """
        Initialize the networks (actor and critic) and their optimizers.
        """

        self.actor = Actor(self.n_observations, self.n_actions, self.env.action_space).to(self.device)
        self.qf1 = SoftQNetwork(self.n_observations, self.n_actions).to(self.device)
        self.qf2 = SoftQNetwork(self.n_observations, self.n_actions).to(self.device)
        self.qf1_target = SoftQNetwork(self.n_observations, self.n_actions).to(self.device)
        self.qf2_target = SoftQNetwork(self.n_observations, self.n_actions).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr)
        self.MSELoss = nn.MSELoss()

    def _init_alpha(self):
        """
        Initialize the entropy coefficient (alpha) and its optimizer.
        """

        self.target_entropy = -torch.prod(
            torch.tensor(self.env.action_space.shape, dtype=torch.float32).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha).item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)

    def _init_vae(self, model_path: str):
        """
        Initialize the VAE model for state representation.

        Args:
            model_path (Optional[str]): Path to the pre-trained VAE model.
        """

        self.vae = ConvVAE(3, 32)
        self.vae.load_state_dict(torch.load(model_path))
        self.vae.eval()

    def _init_rnn(self, model_path: str):
        """
        Initialize the RNN model for state representation.

        Args:
            model_path (str): Path to the pre-trained RNN model.
        """

        self.rnn = MDNRNN(32, 2, 128)
        self.rnn.load_state_dict(torch.load(model_path))
        self.rnn.eval()

    def _init_training_parameters(self):
        """
        Initialize training parameters from the training configuration.
        """

        self.env_id = self.training_config.env_id
        self.n_timesteps = self.training_config.n_timesteps
        self.max_timesteps_per_episode = self.training_config.max_timesteps_per_episode
        self.window_size = self.training_config.window_size
        self.print_freq = self.training_config.print_freq
        self.enable_logging = self.training_config.enable_logging
        self.log_dir = self.training_config.log_dir
        self.checkpoint_freq = self.training_config.checkpoint_freq
        self.save_path = self.training_config.save_path

    def _set_seed(self, seed: int):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Seed value.
        """

        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.device == 'cuda':
            torch.cuda.manual_seed_all(seed)
