#!/usr/bin/env python3
"""Training smoke test for TD3 + PyBullet environment.

Usage:
  python test/test_train_smoke.py
  python test/test_train_smoke.py --steps 600 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test TD3 training loop")
    parser.add_argument("--steps", type=int, default=600, help="Total train steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=20_000, help="Replay buffer size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


def run_smoke(args: argparse.Namespace) -> int:
    ckpt_dir = ROOT / "checkpoints"
    best_ckpt = ckpt_dir / "td3_best.pt"
    last_ckpt = ckpt_dir / "td3_last.pt"

    if best_ckpt.exists():
        best_ckpt.unlink()
    if last_ckpt.exists():
        last_ckpt.unlink()

    train_args = argparse.Namespace(
        seed=args.seed,
        steps=args.steps,
        headless=True,
        cpu=args.cpu,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        start_steps=100,
        explore_noise=0.1,
        explore_noise_high=0.2,
        eval_every=max(200, args.steps // 2),
        eval_episodes=2,
        log_every_step=max(100, args.steps // 3),
        log_every_ep=5,
        eval_only=False,
        checkpoint="td3_best.pt",
    )

    try:
        train(train_args)
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] Training smoke test failed: {exc}")
        return 1

    ok_best = best_ckpt.exists()
    ok_last = last_ckpt.exists()
    if ok_best and ok_last:
        print("[PASS] TD3 training smoke test succeeded")
        print(f"steps={args.steps} batch_size={args.batch_size}")
        print(f"best_ckpt={best_ckpt}")
        print(f"last_ckpt={last_ckpt}")
        return 0

    print("[FAIL] Training completed but checkpoints are missing")
    print(f"best_exists={ok_best} last_exists={ok_last}")
    return 1


def main() -> None:
    args = parse_args()
    code = run_smoke(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
