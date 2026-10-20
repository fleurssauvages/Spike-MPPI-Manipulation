"""FR3 joint-state UDP feedback and joint-position UDP commands.

Input: PC1 -> PC2 UDP 15006, <14d: q[7], dq[7].
Output: PC2 -> PC1 UDP 15007, <7d: desired joint positions.
No Cartesian pose-feedback socket is opened.
"""
from __future__ import annotations
from dataclasses import dataclass
import socket
import struct
import threading
import time
import numpy as np

JOINT = struct.Struct('<14d')
POSITION = struct.Struct('<7d')

@dataclass
class JointSample:
    q: np.ndarray
    dq: np.ndarray
    received_at: float
    sequence: int
    stamp: None = None

class PoseUdpRobot:
    """Compatibility name; only joint feedback is received."""
    def __init__(self, *, pc1='10.20.0.1', pc2='10.20.0.17', enabled=False):
        self.pc1 = pc1
        self.enabled = enabled
        self._lock = threading.Lock()
        self._end = threading.Event()
        self.joint = None
        self.feedback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback.bind((pc2, 15006))
        self.feedback.settimeout(0.05)
        self.command = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command.connect((pc1, 15007))
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self._end.is_set():
            try:
                buf, sender = self.feedback.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if sender[0] != self.pc1 or len(buf) != JOINT.size:
                continue
            values = np.asarray(JOINT.unpack(buf), dtype=float)
            if not np.isfinite(values).all():
                continue
            with self._lock:
                seq = 1 if self.joint is None else self.joint.sequence + 1
                self.joint = JointSample(values[:7].copy(), values[7:].copy(), time.monotonic(), seq)

    def fresh(self, timeout=0.04):
        with self._lock:
            joint = self.joint
        if joint is None or time.monotonic() - joint.received_at > timeout:
            raise RuntimeError('UDP joint feedback on port 15006 missing/stale')
        return joint

    def wait_ready(self, timeout_s=8, max_age=0.04):
        stop = time.monotonic() + timeout_s
        while time.monotonic() < stop:
            try:
                return self.fresh(max_age)
            except RuntimeError:
                time.sleep(0.005)
        raise RuntimeError('Missing UDP joint feedback on port 15006')

    def send_joint_positions(self, target):
        q = np.asarray(target, dtype=float).reshape(7)
        if not np.isfinite(q).all():
            raise RuntimeError('nonfinite joint position command')
        if self.enabled:
            self.command.send(POSITION.pack(*q))

    def close(self):
        self._end.set()
        self.feedback.close()
        self.thread.join(timeout=0.2)
        self.command.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
