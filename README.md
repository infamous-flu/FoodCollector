# Food Collector ML Agents

This repository contains the implementation of an agent-based foraging paradigm in Python, allowing for the manipulation of neural network topology. The software supports different foraging scenarios with varying levels of complexity and is designed to allow for modular, interchangeable neural controllers.

## High Level Research Goal

1. Implement an agent-based foraging paradigm in software allowing for manipulation of neural network topology.

2. Implement different foraging scenarios of various levels of complexity.

## Agents

- `sac.py`: Contains the implementation of the Soft Actor-Critic (SAC) agent.

- `ppo.py`: Contains the implementation of the Proximal Policy Optimization (PPO) agent.

## Environments

- `my_env.py`: Wraps the Unity ML Agents API to a Gymnasium API, making interaction easier by standardizing the step and reset methods.

## Networks

- `mdnrnn.py`: Contains the implementation of the Mixture Density Network Recurrent Neural Network (MDN-RNN).

- `vae.py`: Contains the implementation of the Variational Autoencoder (VAE).

## Running the Agent

To run the the SAC agent, execute the following script:

```bash
python run_sac.py
```

## Training the Agent

To train the SAC agent, first train the VAE by running the following notebook:

```bash
jupyter notebook train_vae.ipynb
```

Then train the MDN-RNN by running the following notebook:

```bash
jupyter notebook train_rnn.ipynb
```

Finally, execute the following script:

```bash
python train_sac.py
```
