#!/usr/bin/env python3
"""Smoke test for PyBulletTrackEnv.

Usage:
  python test/test_env_smoke.py
  python test/test_env_smoke.py --steps 200 --gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pybullet_track_env import PyBulletTrackEnv


def run_smoke_test(steps: int, headless: bool, seed: int) -> int:
    env = PyBulletTrackEnv(headless=headless)
    try:
        obs, _ = env.reset(seed=seed)
        assert obs.shape == env.observation_space.shape, (
            f"Observation shape mismatch: got {obs.shape}, "
            f"expected {env.observation_space.shape}"
        )

        total_reward = 0.0
        term_count = 0
        trunc_count = 0

        for _ in range(steps):
            action = env.action_space.sample().astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            if not np.isfinite(obs).all():
                raise RuntimeError("Observation contains NaN or Inf")
            if not np.isfinite(reward):
                raise RuntimeError("Reward is NaN or Inf")

            total_reward += float(reward)
            term_count += int(terminated)
            trunc_count += int(truncated)

            if terminated or truncated:
                obs, _ = env.reset()

        print("[PASS] PyBulletTrackEnv smoke test succeeded")
        print(f"steps={steps} total_reward={total_reward:.3f} term={term_count} trunc={trunc_count}")
        if "distance" in info:
            print(f"last_distance={float(info['distance']):.3f}")
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] Smoke test failed: {exc}")
        return 1
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test PyBullet tracking environment")
    parser.add_argument("--steps", type=int, default=120, help="Simulation steps to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = run_smoke_test(steps=args.steps, headless=not args.gui, seed=args.seed)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
