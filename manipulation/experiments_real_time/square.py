#!/usr/bin/env python3
"""PC2: trace a 5 cm YZ square using PC1's Cartesian PID controller.

Run `python3 square.py` to preview; `python3 square.py --execute` to move.
UDP pose format: 7 little-endian float64 (x,y,z,qx,qy,qz,qw), metres.
"""
import math
import socket
import struct
import sys
import time

PC1 = "10.20.0.1"   # Franka / PID PC
PC2 = "10.20.0.17"  # Remote PC
COMMAND_PORT = 15005
FEEDBACK_PORT = 15004
SIDE = 0.2             # metres
SPEED = 0.1           # 5 mm/s (reference speed)
HZ = 50
CORNER_TOL = 0.0015    # 1.5 mm
MAX_ERROR = 0.15      # 15 mm tracking error -> abort
MAX_ANGLE = 0.10       # radians orientation error -> abort
FEEDBACK_TIMEOUT = 0.20
SETTLE_TIMEOUT = 12.0  # extra seconds permitted after each edge
POSE = struct.Struct("<7d")


def get_latest(sock):
    """Drain queued datagrams, returning only the latest valid PC1 pose."""
    latest = None
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except BlockingIOError:
            break
        if addr[0] != PC1 or len(data) != POSE.size:
            continue
        p = POSE.unpack(data)
        if not all(math.isfinite(v) for v in p):
            continue
        qnorm = math.sqrt(sum(v*v for v in p[3:]))
        if not 0.95 <= qnorm <= 1.05:
            continue
        latest = (p[:3], tuple(v/qnorm for v in p[3:]))
    return latest


def distance(a, b):
    return math.dist(a, b)


def quat_error(a, b):
    dot = min(1.0, abs(sum(x*y for x, y in zip(a, b))))
    return 2 * math.acos(dot)


def initial_pose(sock):
    """Require >=0.3 s of stable measurements before capturing the start."""
    deadline = time.monotonic() + 6
    samples = []
    while time.monotonic() < deadline:
        pose = get_latest(sock)
        if pose is not None:
            samples.append((time.monotonic(), pose))
            samples = [s for s in samples if time.monotonic() - s[0] < 0.6]
            if len(samples) >= 6 and samples[-1][0] - samples[0][0] >= 0.3:
                first, last = samples[0][1], samples[-1][1]
                if distance(first[0], last[0]) > 0.003 or quat_error(first[1], last[1]) > 0.04:
                    raise RuntimeError("Robot moving: start only from a stable pose")
                return last
        time.sleep(0.01)
    raise RuntimeError("No fresh/stable pose received from PC1 (UDP 15004)")


def main():
    execute = len(sys.argv) == 2 and sys.argv[1] == "--execute"
    if len(sys.argv) > 1 and not execute:
        sys.exit("Usage: python3 square.py [--execute]")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as feedback:
        feedback.bind((PC2, FEEDBACK_PORT))
        feedback.setblocking(False)
        print(f"Waiting for actual TCP pose from {PC1} ...", flush=True)
        (x, y, z), quat = initial_pose(feedback)
        corners = [
            (x, y, z),
            (x, y + SIDE, z),
            (x, y + SIDE, z + SIDE),
            (x, y, z + SIDE),
            (x, y, z),
        ]
        print("YZ square (metres), orientation held constant:")
        for i, p in enumerate(corners):
            print(f"  {i}: {tuple(round(n, 5) for n in p)}")
        print("  quaternion:", quat)
        if not execute:
            print("PREVIEW ONLY. Use --execute to move the robot.")
            return

        period = 1.0 / HZ
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.bind((PC2, 0))
            sender.connect((PC1, COMMAND_PORT))
            last_pose, last_feedback_at = ((x, y, z), quat), time.monotonic()
            for edge in range(4):
                a, b = corners[edge:edge+2]
                edge_started = time.monotonic()
                duration = distance(a, b) / SPEED
                reached_at = None
                while True:
                    t = time.monotonic()
                    pose = get_latest(feedback)
                    if pose is not None:
                        last_pose, last_feedback_at = pose, t
                    if last_pose is None or t - last_feedback_at > FEEDBACK_TIMEOUT:
                        raise RuntimeError("Feedback lost: no more commands will be sent")
                    elapsed = t - edge_started
                    if elapsed > duration + SETTLE_TIMEOUT:
                        raise RuntimeError(f"Side {edge+1} timeout")

                    # Interpolate smoothly along this edge; hold initial orientation.
                    frac = min(1.0, elapsed / duration)
                    target = tuple(v + frac * (w-v) for v, w in zip(a, b))
                    pos, actual_quat = last_pose
                    if distance(pos, target) > MAX_ERROR:
                        raise RuntimeError(f"Tracking error > {MAX_ERROR*1000:g} mm")
                    if quat_error(actual_quat, quat) > MAX_ANGLE:
                        raise RuntimeError("Orientation error too large")
                    sender.send(POSE.pack(*(target + quat)))

                    if (frac == 1.0 and distance(pos, b) < CORNER_TOL
                            and quat_error(actual_quat, quat) < 0.05):
                        if reached_at is None:
                            reached_at = t
                        elif t - reached_at >= 0.3:
                            print(f"Corner {edge+1}/4 reached", flush=True)
                            break
                    else:
                        reached_at = None
                    time.sleep(max(0.0, period - (time.monotonic() - t)))
            print("Square completed; command stream stopped.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, KeyboardInterrupt) as exc:
        sys.exit(f"STOPPED: {exc or 'Ctrl+C'}")
