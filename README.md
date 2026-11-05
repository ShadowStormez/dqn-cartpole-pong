# Deep Q-Network (DQN): CartPole and Pong

This project implements Deep Q-Learning (DQN) and evaluates it on CartPole-v1 and Atari Pong.

## Overview

- Experience replay using a replay buffer.
- Separate online and target Q-networks.
- Epsilon-greedy action selection.
- Smooth L1 loss and gradient clipping.
- MLP architecture for CartPole.
- CNN architecture for Atari Pong.
- Training logs, evaluation results, and learning curves.

## Environments and Results

### CartPole-v1

A simple environment used to debug and validate the DQN implementation.
See `results/cartpole_eval.json` and `results/cartpole_curve.png`.

### Pong

The agent was trained for approximately 2.8 million environment steps.

- Mean reward over 10 evaluation episodes: 18.4
- Standard deviation: 1.43

See `results/pong_eval.json` and `results/pong_curve.png`.

## Repository Structure

```text
dqn-project/
├── dqn.py
├── requirements.txt
├── README.md
├── models/
│   ├── cartpole_dqn.pt
│   └── pong_dqn.pt
└── results/
    ├── cartpole_log.csv
    ├── cartpole_eval.json
    ├── cartpole_curve.png
    ├── pong_log.csv
    ├── pong_eval.json
    └── pong_curve.png
```

## Installation

Python 3.10 or newer is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Atari environments may also require compatible ROM installation and configuration through ALE.

## Implementation

The main implementation is in `dqn.py`.
Experiments were executed in Google Colab.
Trained model weights and experiment outputs are included in this repository.

**Note:** Running `python dqn.py` starts Pong training. It is not necessary to run this command to use the saved results.

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529–533.
  https://doi.org/10.1038/nature14236
- Gymnasium documentation: https://gymnasium.farama.org/
- PyTorch documentation: https://pytorch.org/docs/