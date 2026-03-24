import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: float,
        s2: np.ndarray,
        done: bool,
    ) -> None:
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.d[idx],
        )
