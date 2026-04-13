# Intentional Updates for Streaming Reinforcement Learning

Small PyTorch codebase for the intentional streaming DRL algorithms 

The repository has three runnable training scripts:

- `intentional_ac.py`: Intentional-AC for MuJoCo Gym and DM Control Suite
- `intentional_q_minatar.py`: Intentional-Q for MinAtar
- `intentional_q_atari.py`: Intentional-Q for Atari

Shared utilities:

- `optimizer.py`: custom optimizer
- `normalization_wrappers.py`: observation and reward normalization
- `sparse_init.py`: sparse parameter initialization

## Setup

This repo does not include a dependency file, so setup is manual. You will need Python 3 and the packages imported by the scripts, including:

- `torch`
- `numpy`
- `gymnasium`
- `stable-baselines3`

You will also need environment-specific packages for the tasks you want to run, such as MuJoCo, Atari/ALE, and MinAtar.

Minimal starting point:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy gymnasium stable-baselines3
```

## Run

Continuous control (MuJoCo Gym and DMC):

```bash
python intentional_ac.py --env_name HalfCheetah-v4 --seed 0 --debug
```

Discrete Control (MinAtar):

```bash
python intentional_q_minatar.py --env_name MinAtar/Breakout-v1 --seed 0 --debug
```

Discrete Control (Atari):

```bash
python intentional_q_atari.py --env_name BreakoutNoFrameskip-v4 --seed 0 --debug
```

All scripts also accept `--render`.
