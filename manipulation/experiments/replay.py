from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from manipulation.robots import FrankaReachRobot, FrankaPushRobot, FrankaBallRobot

DEFAULT_REPLAY_LOOKAT = np.asarray([0.48, 0.00, 0.53], dtype=np.float64)
DEFAULT_REPLAY_DISTANCE = 1.30
DEFAULT_REPLAY_AZIMUTH = 145.0
DEFAULT_REPLAY_ELEVATION = -22.0


def _scalar_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return str(value)


def load_recording(path: str | Path) -> dict[str, np.ndarray | str | float]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"robot_name", "qpos", "qvel", "state_time", "controls"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError("recording is missing: " + ", ".join(missing))
        out: dict[str, np.ndarray | str | float] = {k: np.asarray(z[k]).copy() for k in z.files}
        out["robot_name"] = _scalar_text(z["robot_name"])
        out["sampling_option"] = _scalar_text(z["sampling_option"]) if "sampling_option" in z.files else "unknown"
        out["control_dt"] = float(np.asarray(z["control_dt"]).reshape(())) if "control_dt" in z.files else np.nan
    return out


def _prepare_replay(path: str | Path, *, auto_fetch: bool = True):
    record = load_recording(path)
    robot_name = str(record["robot_name"])
    if robot_name == "franka-reach":
        target = np.asarray(record.get("target_xyz", [0.45, -0.25, 0.50]), dtype=np.float64).reshape(3)
        robot = FrankaReachRobot(auto_fetch=auto_fetch, target_xyz=target)
    elif robot_name in {"franka-ball-balancing", "franka-ball-bouncing"}:
        task = robot_name.removeprefix("franka-ball-")
        offset = np.asarray(record.get("initial_ball_offset_xy", [0.0, 0.0]), dtype=np.float64).reshape(2)
        velocity = np.asarray(record.get("initial_ball_velocity_xy", [0.0, 0.0]), dtype=np.float64).reshape(2)
        drop_height = float(np.asarray(record.get("drop_height", 0.23)).reshape(()))
        robot = FrankaBallRobot(
            task=task, auto_fetch=auto_fetch,
            initial_ball_offset_xy=(float(offset[0]), float(offset[1])),
            initial_ball_velocity_xy=(float(velocity[0]), float(velocity[1])),
            drop_height=drop_height,
        )
    elif robot_name == "franka-push":
        start = np.asarray(record.get("start_xyz", [0.55, 0.0, 0.385]), dtype=np.float64).reshape(3)
        goal = np.asarray(record.get("goal_xyz", [0.55, 0.18, 0.385]), dtype=np.float64).reshape(3)
        goal_history = np.asarray(record.get("goal_xyz_history", [goal]), dtype=np.float64)
        if goal_history.ndim == 2 and len(goal_history) and goal_history.shape[1] == 3:
            goal = goal_history[0]
        robot = FrankaPushRobot(auto_fetch=auto_fetch, start_xy=start[:2], goal_xy=goal[:2])
        if "waypoints_xy" in record:
            xy = np.asarray(record["waypoints_xy"], dtype=np.float64)
            if xy.shape == (4, 2):
                robot.set_waypoints_xy(xy)
            elif xy.shape == (2, 2):
                robot.hide_waypoints()
        else:
            robot.hide_waypoints()  # Older point-push recordings had no square sites.
    else:
        raise ValueError(f"unsupported replay robot {robot_name!r}")
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    qvel = np.asarray(record["qvel"], dtype=np.float64)
    state_time = np.asarray(record["state_time"], dtype=np.float64).reshape(-1)
    act = np.asarray(record.get("act", np.zeros((len(qpos), robot.model.na))), dtype=np.float64)
    controls = np.asarray(record["controls"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != robot.model.nq:
        raise ValueError(f"recorded qpos has shape {qpos.shape}, expected (*, {robot.model.nq})")
    if qvel.ndim != 2 or qvel.shape[1] != robot.model.nv:
        raise ValueError(f"recorded qvel has shape {qvel.shape}, expected (*, {robot.model.nv})")
    if len(qpos) != len(qvel) or len(qpos) != len(state_time):
        raise ValueError("qpos, qvel, and state_time must have the same number of frames")
    if robot.model.na and (act.ndim != 2 or act.shape != (len(qpos), robot.model.na)):
        raise ValueError(f"recorded act has shape {act.shape}, expected ({len(qpos)}, {robot.model.na})")
    if controls.ndim != 2 or controls.shape[1] != robot.nu:
        raise ValueError(f"recorded controls have shape {controls.shape}, expected (*, {robot.nu})")
    if "goal_xyz_history" in record:
        hist = np.asarray(record["goal_xyz_history"])
        if hist.shape != (len(qpos), 3):
            raise ValueError(f"goal_xyz_history has shape {hist.shape}, expected ({len(qpos)}, 3)")
    return record, robot, qpos, qvel, state_time, act, controls


def _frame_range(state_time: np.ndarray, start_s: float, end_s: float | None) -> tuple[np.ndarray, int, int]:
    relative_time = state_time - state_time[0]
    first = int(np.searchsorted(relative_time, max(0.0, float(start_s)), side="left"))
    if end_s is None:
        last = len(relative_time) - 1
    else:
        last = int(np.searchsorted(relative_time, max(float(start_s), float(end_s)), side="right") - 1)
        last = min(last, len(relative_time) - 1)
    if first >= len(relative_time) or last < first:
        raise ValueError("requested replay interval contains no recorded frames")
    return relative_time, first, last


def _set_recorded_state(robot, qpos, qvel, state_time, act, controls, i: int,
                        goal_history: np.ndarray | None = None) -> None:
    # The active goal changes at each completed corner; replay the saved mocap
    # target alongside the recorded rigid-body physics state.
    if goal_history is not None and isinstance(robot, FrankaPushRobot):
        goal = np.asarray(goal_history[i], dtype=np.float64)
        if np.linalg.norm(goal - robot.goal_xyz()) > 1e-9:
            robot.set_goal_xy(goal[:2])
    robot.data.time = float(state_time[i])
    robot.data.qpos[:] = qpos[i]
    robot.data.qvel[:] = qvel[i]
    if robot.model.na:
        robot.data.act[:] = act[i]
    if robot.nu:
        if i == 0 or len(controls) == 0:
            robot.data.ctrl[:] = robot.default_ctrl
        else:
            robot.data.ctrl[:] = robot.clip_ctrl(controls[min(i - 1, len(controls) - 1)])
    robot.mujoco.mj_forward(robot.model, robot.data)


def _set_viewer_camera(cam) -> None:
    cam.lookat[:] = DEFAULT_REPLAY_LOOKAT
    cam.distance = DEFAULT_REPLAY_DISTANCE
    cam.azimuth = DEFAULT_REPLAY_AZIMUTH
    cam.elevation = DEFAULT_REPLAY_ELEVATION


def _make_render_camera(mujoco):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    _set_viewer_camera(camera)
    return camera


def export_video(
    path: str | Path, video_path: str | Path, *, format: str = "gif", speed: float = 1.0, fps: float | None = None,
    width: int = 960, height: int = 540, start_s: float = 0.0,
    end_s: float | None = None, auto_fetch: bool = True,
) -> Path:
    try:
        import imageio.v2 as imageio
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Video export requires imageio and MuJoCo (and Pillow for GIF)") from exc
    if format not in {"gif", "mp4"}:
        raise ValueError("format must be gif or mp4")
    if speed <= 0 or width <= 0 or height <= 0:
        raise ValueError("speed and GIF dimensions must be positive")
    record, robot, qpos, qvel, state_time, act, controls = _prepare_replay(path, auto_fetch=auto_fetch)
    relative_time, first, last = _frame_range(state_time, start_s, end_s)
    goal_history = record.get("goal_xyz_history")
    control_dt = float(record.get("control_dt", np.nan))
    if not np.isfinite(control_dt) or control_dt <= 0:
        diffs = np.diff(state_time[first:last + 1])
        diffs = diffs[diffs > 0]
        control_dt = float(np.median(diffs))
    source_fps = 1.0 / control_dt
    video_fps = source_fps if fps is None else float(fps)
    if not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("fps must be positive and finite")
    sim_dt_per_frame = float(speed) / video_fps
    t0, t1 = float(relative_time[first]), float(relative_time[last])
    sample_times = np.arange(t0, t1 + 0.5 * sim_dt_per_frame, sim_dt_per_frame)
    sample_times = np.minimum(sample_times, t1)
    indices = np.searchsorted(relative_time, sample_times, side="left")
    indices = np.clip(indices, first, last)
    prev = np.maximum(first, indices - 1)
    indices = np.where(
        np.abs(relative_time[prev] - sample_times) < np.abs(relative_time[indices] - sample_times),
        prev, indices,
    )
    video_path = Path(video_path).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "mp4" and (width % 2 or height % 2):
        raise ValueError("MP4 width and height must both be even")
    camera = _make_render_camera(mujoco)
    # MuJoCo checks the model's offscreen framebuffer size at Renderer construction.
    # Resize it before creating the renderer, rather than just adjusting output pixels.
    robot.model.vis.global_.offwidth = max(
        int(robot.model.vis.global_.offwidth), int(width)
    )
    robot.model.vis.global_.offheight = max(
        int(robot.model.vis.global_.offheight), int(height)
    )
    # Match the run viewer: use MuJoCo default scene options (including
    # shadows, reflections, geom groups, and site visibility).
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    renderer = mujoco.Renderer(
        robot.model,
        height=height,
        width=width,
        max_geom=max(10000, int(robot.model.ngeom) + 512),
    )
    try:
        if format == "mp4":
            # RGB frames -> H.264/yuv420p, preserving far more colors than GIF.
            # imageio-ffmpeg supplies the ffmpeg executable on most systems.
            writer = imageio.get_writer(
                video_path, format="FFMPEG", mode="I", fps=video_fps,
                codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                output_params=["-crf", "17", "-preset", "medium"],
            )
        else:
            writer = imageio.get_writer(
                video_path, format="GIF-PIL", mode="I", duration=1.0 / video_fps, loop=0
            )
        with writer:
            for i in indices:
                _set_recorded_state(robot, qpos, qvel, state_time, act, controls, int(i), goal_history)
                renderer.update_scene(robot.data, camera=camera, scene_option=scene_option)
                writer.append_data(np.asarray(renderer.render(), dtype=np.uint8))
    finally:
        try:
            renderer.close()
        except Exception:
            pass
    print(f"wrote {format.upper()} {video_path}: frames={len(indices)}, fps={video_fps:g}, size={width}x{height}")
    return video_path


def export_gif(
    path: str | Path, gif_path: str | Path, *, speed: float = 1.0, fps: float | None = None,
    width: int = 960, height: int = 540, start_s: float = 0.0,
    end_s: float | None = None, auto_fetch: bool = True,
) -> Path:
    """Backward-compatible GIF API."""
    return export_video(
        path, gif_path, format="gif", speed=speed, fps=fps,
        width=width, height=height, start_s=start_s, end_s=end_s, auto_fetch=auto_fetch,
    )


def export_mp4(
    path: str | Path, mp4_path: str | Path, *, speed: float = 1.0, fps: float | None = None,
    width: int = 960, height: int = 540, start_s: float = 0.0,
    end_s: float | None = None, auto_fetch: bool = True,
) -> Path:
    """Export replay as high-quality H.264 MP4."""
    return export_video(
        path, mp4_path, format="mp4", speed=speed, fps=fps,
        width=width, height=height, start_s=start_s, end_s=end_s, auto_fetch=auto_fetch,
    )


def replay(
    path: str | Path, *, speed: float = 1.0, loop: bool = False, show_ui: bool = False,
    start_s: float = 0.0, end_s: float | None = None, pause_at_end: float = 0.25,
    auto_fetch: bool = True,
) -> None:
    record, robot, qpos, qvel, state_time, act, controls = _prepare_replay(path, auto_fetch=auto_fetch)
    relative_time, first, last = _frame_range(state_time, start_s, end_s)
    goal_history = record.get("goal_xyz_history")
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(
        robot.model, robot.data, show_left_ui=bool(show_ui), show_right_ui=bool(show_ui)
    )
    _set_viewer_camera(viewer.cam)
    viewer.sync()
    try:
        print(
            f"replay {Path(path).name}: sampling={record['sampling_option']}, "
            f"frames={last-first+1}, speed={speed:g}x"
        )
        keep_running = True
        while keep_running and viewer.is_running():
            wall0 = time.perf_counter()
            sim0 = float(relative_time[first])
            for i in range(first, last + 1):
                if not viewer.is_running():
                    keep_running = False
                    break
                wait = wall0 + (float(relative_time[i]) - sim0) / float(speed) - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                _set_recorded_state(robot, qpos, qvel, state_time, act, controls, i, goal_history)
                viewer.sync()
            if not loop:
                break
            if keep_running and pause_at_end > 0:
                time.sleep(float(pause_at_end))
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        time.sleep(0.20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved Franka reach benchmark run")
    parser.add_argument("--file", default="manipulation/results/last_run.npz")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--viewer-ui", action="store_true")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--pause-at-end", type=float, default=0.25)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--gif", default=None)
    output.add_argument("--mp4", default=None)
    parser.add_argument("--fps", "--gif-fps", "--mp4-fps", dest="output_fps", type=float, default=None)
    parser.add_argument("--width", "--gif-width", "--mp4-width", dest="output_width", type=int, default=960)
    parser.add_argument("--height", "--gif-height", "--mp4-height", dest="output_height", type=int, default=540)
    parser.add_argument("--no-auto-fetch", action="store_true")
    args = parser.parse_args()
    if args.gif or args.mp4:
        export_video(
            args.file, args.mp4 if args.mp4 else args.gif,
            format="mp4" if args.mp4 else "gif", speed=args.speed, fps=args.output_fps,
            width=args.output_width, height=args.output_height, start_s=args.start,
            end_s=args.end, auto_fetch=not args.no_auto_fetch,
        )
    else:
        replay(
            args.file, speed=args.speed, loop=args.loop, show_ui=args.viewer_ui,
            start_s=args.start, end_s=args.end, pause_at_end=args.pause_at_end,
            auto_fetch=not args.no_auto_fetch,
        )


if __name__ == "__main__":
    main()
