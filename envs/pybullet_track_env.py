"""Gymnasium environment: PyBullet racecar reaches a planar target position."""

from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

try:
    import pybullet_data
except ModuleNotFoundError:  # pragma: no cover
    pybullet_data = None  # type: ignore[assignment]


def _joint_index_by_name(body_id: int, name: str) -> int:
    for i in range(p.getNumJoints(body_id)):
        jn = p.getJointInfo(body_id, i)[1]
        if isinstance(jn, bytes):
            jn = jn.decode("utf-8")
        if jn == name:
            return i
    raise ValueError(f"Joint {name!r} not found on body {body_id}")


def _joint_index_by_any_name(body_id: int, candidates: list[str]) -> int:
    """Return first joint index matching any of `candidates`."""
    last_err: Optional[ValueError] = None
    for name in candidates:
        try:
            return _joint_index_by_name(body_id, name)
        except ValueError as e:
            last_err = e
    assert last_err is not None
    raise last_err


class PyBulletTrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        headless: bool = True,
        max_episode_steps: int = 500,
        arena: float = 6.0,
        max_wheel_vel: float = 18.0,
        reach_threshold: float = 0.45,
        urdf_subpath: str = "racecar/racecar.urdf",
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.headless = headless
        self.max_episode_steps = max_episode_steps
        self.arena_half = arena / 2.0
        self.max_wheel_vel = max_wheel_vel
        self.reach_threshold = reach_threshold
        self.urdf_subpath = urdf_subpath

        self._client: Optional[int] = None
        self._plane_id: Optional[int] = None
        self._car_id: Optional[int] = None
        self._target_id: Optional[int] = None
        self._steering_joints: list[int] = []
        self._drive_joints: tuple[int, int] = (0, 0)

        self._step_count = 0
        self._target_xy = np.zeros(2, dtype=np.float32)

        # obs: rel_x, rel_y, dist, vx, vy, wz (world frame), scaled
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._connect_and_load()

    def _connect_and_load(self) -> None:
        if self._client is not None:
            return
        mode = p.DIRECT if self.headless else p.GUI
        self._client = p.connect(mode)
        # `pybullet.getDataPath()` 在部分版本里不存在。
        if hasattr(p, "getDataPath"):
            p.setAdditionalSearchPath(p.getDataPath())  # type: ignore[attr-defined]
        else:
            if pybullet_data is None:
                raise ModuleNotFoundError(
                    "pybullet_data is required for loading URDFs like plane.urdf/racecar.urdf"
                )
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        start_pos = [0.0, 0.0, 0.1]
        start_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self._car_id = p.loadURDF(
            self.urdf_subpath, start_pos, start_orn, flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        assert self._car_id is not None

        # Drive rear wheels; center steering for differential-style control.
        for jn_candidates in [
            ["left_steering_hinge_joint", "left_steering_hinge"],
            ["right_steering_hinge_joint", "right_steering_hinge"],
        ]:
            try:
                self._steering_joints.append(
                    _joint_index_by_any_name(self._car_id, jn_candidates)
                )
            except ValueError:
                pass
        # Prefer rear wheels for racecar (PyBullet's default URDF).
        self._drive_joints = (
            _joint_index_by_any_name(
                self._car_id,
                ["left_rear_wheel_joint", "left_wheel_joint", "left_wheel"],
            ),
            _joint_index_by_any_name(
                self._car_id,
                ["right_rear_wheel_joint", "right_wheel_joint", "right_wheel"],
            ),
        )

        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.18, rgbaColor=[0.1, 0.85, 0.2, 0.9])
        self._target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs,
            basePosition=[0.0, 0.0, 0.05],
        )
        p.setTimeStep(1.0 / 240.0)

    def _apply_action(self, action: np.ndarray) -> None:
        assert self._car_id is not None
        a = np.clip(action.astype(np.float64), -1.0, 1.0)
        left_v = float(a[0] * self.max_wheel_vel)
        right_v = float(a[1] * self.max_wheel_vel)
        for sj in self._steering_joints:
            p.setJointMotorControl2(
                self._car_id,
                sj,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=8.0,
            )
        p.setJointMotorControl2(
            self._car_id,
            self._drive_joints[0],
            p.VELOCITY_CONTROL,
            targetVelocity=left_v,
            force=25.0,
        )
        p.setJointMotorControl2(
            self._car_id,
            self._drive_joints[1],
            p.VELOCITY_CONTROL,
            targetVelocity=right_v,
            force=25.0,
        )

    def _sim_substeps(self, n: int = 12) -> None:
        for _ in range(n):
            p.stepSimulation()

    def _get_obs(self) -> np.ndarray:
        assert self._car_id is not None and self._target_id is not None
        pos, orn = p.getBasePositionAndOrientation(self._car_id)
        lin, ang = p.getBaseVelocity(self._car_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        rx, ry = float(pos[0]), float(pos[1])
        tx, ty = float(self._target_xy[0]), float(self._target_xy[1])
        rel_x = tx - rx
        rel_y = ty - ry
        dist = math.hypot(rel_x, rel_y)
        vx, vy = float(lin[0]), float(lin[1])
        wz = float(ang[2])
        scale_p = self.arena_half + 0.5
        scale_v = self.max_wheel_vel * 0.35
        obs = np.array(
            [
                rel_x / scale_p,
                rel_y / scale_p,
                dist / scale_p,
                vx / scale_v,
                vy / scale_v,
                wz / 4.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _distance(self) -> float:
        assert self._car_id is not None
        pos, _ = p.getBasePositionAndOrientation(self._car_id)
        dx = self._target_xy[0] - pos[0]
        dy = self._target_xy[1] - pos[1]
        return float(math.hypot(dx, dy))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        assert self._car_id is not None and self._target_id is not None

        margin = 0.8
        low = -self.arena_half + margin
        high = self.arena_half - margin
        rx = self.np_random.uniform(low, high)
        ry = self.np_random.uniform(low, high)
        yaw = self.np_random.uniform(-math.pi, math.pi)
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(self._car_id, [rx, ry, 0.1], orn)
        p.resetBaseVelocity(self._car_id, [0, 0, 0], [0, 0, 0])

        for _ in range(20):
            for sj in self._steering_joints:
                p.setJointMotorControl2(
                    self._car_id, sj, p.POSITION_CONTROL, targetPosition=0.0, force=8.0
                )
            p.setJointMotorControl2(
                self._car_id,
                self._drive_joints[0],
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=25.0,
            )
            p.setJointMotorControl2(
                self._car_id,
                self._drive_joints[1],
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=25.0,
            )
            p.stepSimulation()

        min_sep = 1.2
        for _ in range(50):
            tx = self.np_random.uniform(low, high)
            ty = self.np_random.uniform(low, high)
            if math.hypot(tx - rx, ty - ry) >= min_sep:
                break
        self._target_xy[0] = tx
        self._target_xy[1] = ty
        p.resetBasePositionAndOrientation(self._target_id, [tx, ty, 0.05], [0, 0, 0, 1])

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._apply_action(np.asarray(action, dtype=np.float32))
        self._sim_substeps(12)
        self._step_count += 1

        dist = self._distance()
        obs = self._get_obs()

        # Dense reward + small action penalty; bonus on success.
        reward = -0.4 * dist
        reward -= 0.02 * float(np.sum(np.square(action)))
        success = dist < self.reach_threshold
        if success:
            reward += 15.0

        terminated = success
        truncated = self._step_count >= self.max_episode_steps

        return obs, float(reward), terminated, truncated, {"distance": dist, "success": success}

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            w, h, rgb, _, _ = p.getCameraImage(
                320,
                240,
                viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=8.0,
                    yaw=45,
                    pitch=-35,
                    roll=0,
                    upAxisIndex=2,
                ),
                projectionMatrix=p.computeProjectionMatrixFOV(60, 320 / 240, 0.1, 100.0),
                renderer=p.ER_BULLET_HARDWARE_OPENGL
                if not self.headless
                else p.ER_TINY_RENDERER,
            )
            return np.reshape(rgb, (h, w, 4))[:, :, :3]
        return None

    def close(self) -> None:
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None
            self._car_id = None
            self._target_id = None
