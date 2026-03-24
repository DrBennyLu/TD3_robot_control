# TD3 robot control (PyBullet)

Twin Delayed DDPG (TD3) trains a **PyBullet built-in racecar** to reach a random planar target. The stack is **Gymnasium + PyTorch + PyBullet** and is intended to run the same way on **macOS** and **Windows** (venv + pip, no hardcoded paths).

## Requirements

- **Python 3.10–3.12** recommended (CPU wheels for `torch` are widely available).
- **Python 3.13+**: install `torch` from [pytorch.org](https://pytorch.org/) for your OS (CPU or CUDA), then install the rest from `requirements.txt`.

## Setup (Mac / Windows)

```bash
cd TD3_robot_control
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows (cmd): .venv\Scripts\activate.bat
# Windows (PowerShell): .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

URDF and plane assets come from PyBullet’s data path (`pybullet.getDataPath()`); no separate `pybullet_data` package is required.

## Train

Headless (no GUI, good for servers / CI):

```bash
python train.py --headless --steps 80000 --seed 0
```

With GUI (local machine with a display):

```bash
python train.py --steps 80000 --seed 0
```

Checkpoints are written to `checkpoints/td3_best.pt` (best eval return) and `checkpoints/td3_last.pt`.

### Quick smoke test

```bash
python train.py --headless --steps 1500 --eval-every 500 --eval-episodes 2 --log-every-step 500
```

You should see `critic` / `actor` log lines and periodic `eval` lines without errors.

### Smoke test scripts

Environment-only smoke test (no TD3 training):

```bash
python test/test_env_smoke.py
python test/test_env_smoke.py --steps 200 --gui
```

Training-loop smoke test (short TD3 run + checkpoint check):

```bash
python test/test_train_smoke.py
python test/test_train_smoke.py --steps 800 --batch-size 64
```

Expected result: both scripts print `[PASS]`.

## Evaluate a saved policy

```bash
python train.py --eval-only --checkpoint checkpoints/td3_best.pt --eval-episodes 10 --headless
```

Omit `--headless` to watch the racecar in the PyBullet GUI.

## Cross-platform checklist

1. Create a fresh venv on each OS.
2. `pip install -r requirements.txt` (or install `torch` separately on 3.13+ as above).
3. Run `python test/test_env_smoke.py` and `python test/test_train_smoke.py`.

## Project layout

| Path | Role |
|------|------|
| `envs/pybullet_track_env.py` | Gymnasium env: racecar, target sphere, reward, reset |
| `td3/networks.py` | Actor + twin critics |
| `td3/replay_buffer.py` | Replay buffer |
| `td3/td3_agent.py` | TD3 updates, save/load |
| `train.py` | Training and evaluation CLI |

## Author

Benny Lu
