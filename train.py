#!/usr/bin/env python3
"""Train or evaluate TD3 on PyBullet planar position tracking."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pybullet_track_env import PyBulletTrackEnv
from td3.replay_buffer import ReplayBuffer
from td3.td3_agent import TD3Agent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    env: PyBulletTrackEnv,
    agent: TD3Agent,
    episodes: int,
    *,
    seed: int,
) -> tuple[float, float]:
    set_seed(seed)
    returns: list[float] = []
    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        term = trunc = False
        while not (term or trunc):
            action = agent.select_action(obs, explore_noise=0.0)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += r
        returns.append(ep_ret)
        successes += int(term)
    return float(np.mean(returns)), successes / max(episodes, 1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    env = PyBulletTrackEnv(headless=args.headless)
    eval_env = PyBulletTrackEnv(headless=args.headless)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(
        state_dim,
        action_dim,
        max_action,
        device=device,
    )
    buffer = ReplayBuffer(state_dim, action_dim, capacity=args.buffer_size)

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "td3_best.pt"
    last_path = ckpt_dir / "td3_last.pt"

    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    ep_num = 0
    best_eval = -1e18

    for t in range(1, args.steps + 1):
        explore = args.explore_noise if t > args.start_steps else args.explore_noise_high
        action = agent.select_action(obs, explore_noise=explore)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, float(terminated))

        ep_reward += reward
        if done:
            ep_num += 1
            if ep_num % args.log_every_ep == 0:
                print(f"episode {ep_num} step {t} ep_return {ep_reward:.2f}")
            ep_reward = 0.0
            obs, _ = env.reset(seed=args.seed + t)
        else:
            obs = next_obs

        if buffer.size >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            losses = agent.update(batch)
            if t % args.log_every_step == 0:
                print(
                    f"step {t} critic {losses['critic_loss']:.4f} actor {losses['actor_loss']:.4f}"
                )

        if t % args.eval_every == 0 and t > 0:
            mean_ret, succ_rate = evaluate(eval_env, agent, args.eval_episodes, seed=args.seed + 999)
            print(f"eval @ {t}: mean_return {mean_ret:.2f} success_rate {succ_rate:.2f}")
            if mean_ret > best_eval:
                best_eval = mean_ret
                agent.save(str(best_path))
            agent.save(str(last_path))

    env.close()
    eval_env.close()
    print(f"Training finished. Best checkpoint: {best_path}")


def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = PyBulletTrackEnv(headless=args.headless)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3Agent(state_dim, action_dim, max_action, device=device)
    path = Path(args.checkpoint)
    if not path.is_file():
        path = ROOT / "checkpoints" / args.checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    agent.load(str(path))
    mean_ret, succ_rate = evaluate(env, agent, args.eval_episodes, seed=args.seed)
    print(f"Eval mean_return {mean_ret:.2f} success_rate {succ_rate:.2f}")
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TD3 PyBullet position tracking")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--headless", action="store_true", help="PyBullet DIRECT (no GUI)")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--start-steps", type=int, default=5_000, help="Higher exploration noise until this many steps")
    p.add_argument("--explore-noise", type=float, default=0.1)
    p.add_argument("--explore-noise-high", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=4_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--log-every-step", type=int, default=2_000)
    p.add_argument("--log-every-ep", type=int, default=10)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default="td3_best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_only:
        run_eval(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
