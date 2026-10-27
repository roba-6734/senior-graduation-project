#!/usr/bin/env python3
"""Evaluate a Unitree G1 tracking checkpoint over one complete motion.

This script is intentionally kept outside the Unitree repository.  It imports
the pinned Unitree RL MjLab checkout at runtime, initializes the robot exactly
at reference frame zero, disables randomization and automatic resets, and then
measures reference-to-simulation tracking for every remaining frame.

The resulting tracking error is not the same quantity as GMR's human-to-robot
retargeting error.  Both are retained in the JSON report when a GMR evaluation
report is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_UNITREE_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
TASK_ID = "Unitree-G1-Tracking-No-State-Estimation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def get_retarget_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    smoothed = payload["smoothed"]
    return {
        "source_report": str(path.resolve()),
        "status": payload["status"],
        "metric_scope": "human SMPL-X to kinematic Unitree G1 reference",
        "mean_matched_body_position_error_m": smoothed["fidelity"][
            "all_position_error_m"
        ]["mean"],
        "mean_upper_body_position_error_m": smoothed["fidelity"][
            "upper_body_position_error_m"
        ]["mean"],
        "max_motor_acceleration_rad_s2": smoothed["feasibility"][
            "motor_acceleration_rad_s2"
        ]["max"],
    }


def render_frame(env: Any) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f"Unexpected rendered frame shape: {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def actuator_force_limits(robot: Any, torch: Any) -> Any:
    limits = torch.full(
        (robot.data.actuator_force.shape[0], robot.num_actuators),
        float("nan"),
        device=robot.data.actuator_force.device,
    )
    for actuator in robot.actuators:
        force_limit = getattr(actuator, "force_limit", None)
        if force_limit is None:
            effort_limit = getattr(actuator.cfg, "effort_limit", None)
            if effort_limit is not None:
                force_limit = torch.full(
                    (limits.shape[0], len(actuator.ctrl_ids)),
                    float(effort_limit),
                    device=limits.device,
                )
        if force_limit is not None:
            limits[:, actuator.ctrl_ids] = force_limit
    return limits


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    tracking = summary["reference_to_simulation_tracking"]
    stability = summary["stability"]
    lines = [
        "# Ayyala dynamic tracking evaluation",
        "",
        f"- Complete reference survived: **{stability['survived_full_motion']}**",
        f"- Evaluated frames: **{summary['evaluated_frames']} / {summary['expected_evaluated_frames']}**",
        f"- Reference rate: **{summary['reference_fps_hz']:.3f} Hz**",
        f"- Mean MPKPE: **{100.0 * tracking['mpkpe_m']['mean']:.2f} cm**",
        f"- Mean root-relative MPKPE: **{100.0 * tracking['root_relative_mpkpe_m']['mean']:.2f} cm**",
        f"- Mean end-effector position error: **{100.0 * tracking['end_effector_position_error_m']['mean']:.2f} cm**",
        f"- Mean joint-position MAE: **{tracking['joint_position_mae_rad']['mean']:.4f} rad**",
        f"- Peak actuator force: **{tracking['actuator_abs_force_nm']['max']:.2f} N·m**",
        f"- Actuator saturation sample fraction: **{100.0 * tracking['actuator_saturation_fraction']:.3f}%**",
        f"- Joint-limit violation samples: **{stability['joint_limit_violation_samples']}**",
        "",
        "MPKPE here measures the trained policy against the robot reference in dynamic simulation. "
        "It must not be merged with the human-to-robot GMR retargeting error.",
    ]
    retarget = summary.get("human_to_robot_retargeting")
    if retarget is not None:
        lines.extend(
            [
                "",
                "## Separate GMR retargeting result",
                "",
                "- Mean matched-body error: "
                f"**{100.0 * retarget['mean_matched_body_position_error_m']:.2f} cm**",
                "- Mean upper-body error: "
                f"**{100.0 * retarget['mean_upper_body_position_error_m']:.2f} cm**",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    motion_file = args.motion_file.resolve()
    checkpoint_file = args.checkpoint_file.resolve()
    output_dir = args.output_dir.resolve()
    video_file = args.video_file.resolve() if args.video_file is not None else None
    retarget_report = (
        args.retarget_report.resolve() if args.retarget_report is not None else None
    )

    for required in (repo / "scripts" / "train.py", motion_file, checkpoint_file):
        if not required.exists():
            raise FileNotFoundError(required)
    current_commit = git_commit(repo)
    if not args.allow_commit_mismatch and current_commit != EXPECTED_UNITREE_COMMIT:
        raise RuntimeError(
            f"Unitree checkout is {current_commit}, but this evaluator targets "
            f"{EXPECTED_UNITREE_COMMIT}. Pass --allow-commit-mismatch only after reviewing "
            "upstream API changes."
        )

    # Unitree's scripts rely on the checkout root being both cwd and sys.path.
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import mediapy as media
    import torch

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from mjlab.tasks.tracking.mdp.metrics import (
        compute_ee_orientation_error,
        compute_ee_position_error,
        compute_joint_velocity_error,
        compute_mpkpe,
        compute_root_relative_mpkpe,
    )
    from mjlab.utils.lab_api.math import (
        quat_apply_inverse,
        quat_error_magnitude,
    )
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA evaluation requested, but torch.cuda.is_available() is false"
        )

    with np.load(motion_file, allow_pickle=False) as motion_archive:
        reference_frames = int(motion_archive["joint_pos"].shape[0])
        reference_fps = float(
            np.asarray(motion_archive.get("fps", [50.0])).reshape(-1)[0]
        )
    if reference_frames < 2:
        raise ValueError("Reference motion needs at least two frames")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    motion_cfg = env_cfg.commands.get("motion")
    if not isinstance(motion_cfg, MotionCommandCfg):
        raise ValueError(f"Task {args.task} is not a motion-tracking task")

    motion_cfg.motion_file = str(motion_file)
    motion_cfg.sampling_mode = "start"
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.debug_vis = video_file is not None
    motion_cfg.viz.mode = "ghost"

    ee_body_names = tuple(env_cfg.terminations["ee_body_pos"].params["body_names"])
    # Automatic resets destroy the failed state before it can be logged.  Run
    # the full sequence and compute the same three failure conditions ourselves.
    env_cfg.terminations = {}
    env_cfg.events = {}
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = reference_frames / reference_fps + 1.0
    env_cfg.viewer.height = args.video_height
    env_cfg.viewer.width = args.video_width

    output_dir.mkdir(parents=True, exist_ok=True)
    if video_file is not None:
        video_file.parent.mkdir(parents=True, exist_ok=True)

    render_mode = "rgb_array" if video_file is not None else None
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint_file),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    command = base_env.command_manager.get_term("motion")
    robot = base_env.scene["robot"]
    ee_indices = [
        index for index, name in enumerate(command.cfg.body_names) if name in ee_body_names
    ]
    if len(ee_indices) != len(ee_body_names):
        raise RuntimeError(
            f"Could not resolve all end effectors {ee_body_names}; resolved indices {ee_indices}"
        )
    force_limits = actuator_force_limits(robot, torch)

    series: dict[str, list[float | int | bool]] = {
        "reference_frame": [],
        "time_s": [],
        "mpkpe_m": [],
        "root_relative_mpkpe_m": [],
        "end_effector_position_error_m": [],
        "end_effector_orientation_error_rad": [],
        "root_position_error_m": [],
        "root_orientation_error_rad": [],
        "joint_position_l2_rad": [],
        "joint_position_mae_rad": [],
        "joint_velocity_l2_rad_s": [],
        "actuator_abs_force_nm": [],
        "actuator_saturation_fraction": [],
        "action_clip_fraction": [],
        "joint_limit_violation_count": [],
        "failure_anchor_height": [],
        "failure_anchor_orientation": [],
        "failure_end_effector_height": [],
    }

    obs = env.get_observations()
    video_writer: Any = None
    first_failure_frame: int | None = None
    numerical_failure: str | None = None
    expected_steps = reference_frames - 1
    clip_actions = agent_cfg.clip_actions

    try:
        if video_file is not None:
            frame = render_frame(base_env)
            video_writer = media.VideoWriter(
                video_file,
                frame.shape[:2],
                fps=1.0 / base_env.step_dt,
                crf=20,
            )
            video_writer.__enter__()
            video_writer.add_image(frame)

        for step in range(expected_steps):
            with torch.inference_mode():
                actions = policy(obs)
                if clip_actions is None:
                    action_clip_fraction = 0.0
                else:
                    action_clip_fraction = float(
                        (torch.abs(actions) > float(clip_actions)).float().mean().item()
                    )
                obs, _, _, _ = env.step(actions)

            frame_index = int(command.time_steps[0].item())
            expected_frame_index = step + 1
            if frame_index != expected_frame_index:
                raise RuntimeError(
                    f"Reference advanced to frame {frame_index}, expected {expected_frame_index}"
                )

            joint_delta = command.joint_pos - command.robot_joint_pos
            root_pos_error = torch.linalg.norm(
                command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1
            )
            root_ori_error = quat_error_magnitude(
                command.anchor_quat_w, command.robot_anchor_quat_w
            )
            forces = torch.abs(robot.data.actuator_force)
            valid_force_limits = torch.isfinite(force_limits) & (force_limits > 0.0)
            saturation = torch.where(
                valid_force_limits,
                forces >= (0.99 * force_limits),
                torch.zeros_like(valid_force_limits),
            )
            limits = robot.data.joint_pos_limits
            joint_limit_violations = (robot.data.joint_pos < limits[..., 0]) | (
                robot.data.joint_pos > limits[..., 1]
            )

            anchor_height_failure = torch.abs(
                command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
            ) > 0.25
            motion_gravity = quat_apply_inverse(
                command.anchor_quat_w, robot.data.gravity_vec_w
            )
            robot_gravity = quat_apply_inverse(
                command.robot_anchor_quat_w, robot.data.gravity_vec_w
            )
            anchor_orientation_failure = torch.abs(
                motion_gravity[:, 2] - robot_gravity[:, 2]
            ) > 0.8
            ee_height_error = torch.abs(
                command.body_pos_relative_w[:, ee_indices, 2]
                - command.robot_body_pos_w[:, ee_indices, 2]
            )
            ee_height_failure = torch.any(ee_height_error > 0.25, dim=-1)

            values = {
                "reference_frame": frame_index,
                "time_s": frame_index / reference_fps,
                "mpkpe_m": float(compute_mpkpe(command)[0].item()),
                "root_relative_mpkpe_m": float(
                    compute_root_relative_mpkpe(command)[0].item()
                ),
                "end_effector_position_error_m": float(
                    compute_ee_position_error(command, ee_body_names)[0].item()
                ),
                "end_effector_orientation_error_rad": float(
                    compute_ee_orientation_error(command, ee_body_names)[0].item()
                ),
                "root_position_error_m": float(root_pos_error[0].item()),
                "root_orientation_error_rad": float(root_ori_error[0].item()),
                "joint_position_l2_rad": float(
                    torch.linalg.norm(joint_delta, dim=-1)[0].item()
                ),
                "joint_position_mae_rad": float(
                    torch.mean(torch.abs(joint_delta), dim=-1)[0].item()
                ),
                "joint_velocity_l2_rad_s": float(
                    compute_joint_velocity_error(command)[0].item()
                ),
                "actuator_abs_force_nm": float(torch.max(forces[0]).item()),
                "actuator_saturation_fraction": float(
                    saturation[0].float().mean().item()
                ),
                "action_clip_fraction": action_clip_fraction,
                "joint_limit_violation_count": int(
                    joint_limit_violations[0].sum().item()
                ),
                "failure_anchor_height": bool(anchor_height_failure[0].item()),
                "failure_anchor_orientation": bool(anchor_orientation_failure[0].item()),
                "failure_end_effector_height": bool(ee_height_failure[0].item()),
            }
            numeric_values = [
                value
                for value in values.values()
                if isinstance(value, (float, np.floating))
            ]
            if not np.isfinite(numeric_values).all():
                numerical_failure = f"Non-finite value at reference frame {frame_index}"
                break
            for key, value in values.items():
                series[key].append(value)

            if first_failure_frame is None and (
                values["failure_anchor_height"]
                or values["failure_anchor_orientation"]
                or values["failure_end_effector_height"]
            ):
                first_failure_frame = frame_index

            if video_writer is not None:
                video_writer.add_image(render_frame(base_env))
    finally:
        if video_writer is not None:
            video_writer.__exit__(None, None, None)
        env.close()

    arrays = {key: np.asarray(value) for key, value in series.items()}
    np.savez_compressed(output_dir / "tracking_timeseries.npz", **arrays)

    evaluated_frames = len(series["reference_frame"])
    metric_keys = (
        "mpkpe_m",
        "root_relative_mpkpe_m",
        "end_effector_position_error_m",
        "end_effector_orientation_error_rad",
        "root_position_error_m",
        "root_orientation_error_rad",
        "joint_position_l2_rad",
        "joint_position_mae_rad",
        "joint_velocity_l2_rad_s",
        "actuator_abs_force_nm",
        "action_clip_fraction",
    )
    tracking_metrics = {key: stats(arrays[key]) for key in metric_keys}
    tracking_metrics["actuator_saturation_fraction"] = float(
        np.mean(arrays["actuator_saturation_fraction"])
    ) if evaluated_frames else float("nan")

    summary: dict[str, Any] = {
        "status": "COMPLETE" if evaluated_frames == expected_steps else "INCOMPLETE",
        "metric_scope": "trained Unitree policy to Unitree reference in dynamic MuJoCo simulation",
        "task": args.task,
        "reference_frames": reference_frames,
        "expected_evaluated_frames": expected_steps,
        "evaluated_frames": evaluated_frames,
        "reference_fps_hz": reference_fps,
        "reference_to_simulation_tracking": tracking_metrics,
        "stability": {
            "survived_full_motion": bool(
                evaluated_frames == expected_steps
                and first_failure_frame is None
                and numerical_failure is None
            ),
            "first_termination_equivalent_failure_frame": first_failure_frame,
            "numerical_failure": numerical_failure,
            "anchor_height_failure_frames": int(
                np.count_nonzero(arrays["failure_anchor_height"])
            ),
            "anchor_orientation_failure_frames": int(
                np.count_nonzero(arrays["failure_anchor_orientation"])
            ),
            "end_effector_height_failure_frames": int(
                np.count_nonzero(arrays["failure_end_effector_height"])
            ),
            "joint_limit_violation_samples": int(
                np.sum(arrays["joint_limit_violation_count"])
            ),
        },
        "artifacts": {
            "timeseries_npz": str((output_dir / "tracking_timeseries.npz").resolve()),
            "video": str(video_file) if video_file is not None else None,
        },
        "provenance": {
            "unitree_rl_mjlab_repo": str(repo),
            "unitree_rl_mjlab_commit": current_commit,
            "motion_file": str(motion_file),
            "motion_sha256": sha256_file(motion_file),
            "checkpoint_file": str(checkpoint_file),
            "checkpoint_sha256": sha256_file(checkpoint_file),
            "seed": args.seed,
            "device": args.device,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
        "human_to_robot_retargeting": get_retarget_summary(retarget_report),
    }

    summary_path = output_dir / "tracking_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(output_dir / "tracking_metrics.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved evaluation to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retarget-report", type=Path)
    parser.add_argument("--video-file", type=Path)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-commit-mismatch", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
