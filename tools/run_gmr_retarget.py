#!/usr/bin/env python3
"""Run a headless GMR retarget and evaluate the resulting G1 reference motion."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import smplx
import torch

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast


def load_human_frames(motion_path: Path, model_root: Path) -> tuple[list[dict], float, float]:
    """Load SMPL-X with explicit dimensions required by current smplx releases."""
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    frame_count = len(motion["pose_body"])
    betas = np.asarray(motion["betas"], dtype=np.float32).reshape(-1)
    body_model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender=str(np.asarray(motion["gender"]).item()),
        use_pca=False,
        num_betas=len(betas),
        batch_size=frame_count,
    )
    body_model.eval()
    with torch.no_grad():
        output = body_model(
            betas=torch.from_numpy(betas).view(1, -1),
            global_orient=torch.from_numpy(motion["root_orient"]),
            body_pose=torch.from_numpy(motion["pose_body"]),
            transl=torch.from_numpy(motion["trans"]),
            left_hand_pose=torch.zeros(frame_count, 45),
            right_hand_pose=torch.zeros(frame_count, 45),
            jaw_pose=torch.zeros(frame_count, 3),
            leye_pose=torch.zeros(frame_count, 3),
            reye_pose=torch.zeros(frame_count, 3),
            expression=torch.zeros(frame_count, 10),
            return_full_pose=True,
        )
    frames, fps = get_smplx_data_offline_fast(
        motion, body_model, output, tgt_fps=30
    )
    human_height = float(1.66 + 0.1 * betas[0])
    return frames, float(fps), human_height


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def quaternion_speed(quaternions: np.ndarray, fps: float) -> np.ndarray:
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    dots = np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0)) * fps


def summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


def contact_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def render_motion(model_path: str, qpos: np.ndarray, fps: float, output_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=720)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.8
    camera.azimuth = 135
    camera.elevation = -12
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for configuration in qpos:
            data.qpos[:] = configuration
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.xpos[model.body("pelvis").id]
            camera.lookat[2] = 0.75
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def markdown_report(report: dict[str, Any]) -> str:
    fidelity = report["fidelity"]
    feasibility = report["feasibility"]
    reasons = report.get("review_reasons", [])
    reasons_markdown = "\n".join(f"- {reason}" for reason in reasons)
    if reasons_markdown:
        reasons_markdown = f"\n## Review reasons\n\n{reasons_markdown}\n"
    return f"""# GMR Unitree G1 29-DoF evaluation

**Status:** {report['status']}

- Frames: {report['frames']} at {report['fps_hz']:.3g} FPS
- Processing rate: {report['processing_fps']:.1f} FPS
- Mean matched-body position error: {fidelity['all_position_error_m']['mean'] * 100:.2f} cm
- Mean upper-body position error: {fidelity['upper_body_position_error_m']['mean'] * 100:.2f} cm
- Mean lower-body position error: {fidelity['lower_body_position_error_m']['mean'] * 100:.2f} cm
- Joint-limit violations: {feasibility['joint_limit_violation_count']}
- Velocity-limit violations: {feasibility['velocity_limit_violation_count']}
- Maximum motor speed: {feasibility['motor_speed_rad_s']['max']:.3f} rad/s
- Maximum motor acceleration: {feasibility['motor_acceleration_rad_s2']['max']:.3f} rad/s^2
- Frames over the diagnostic acceleration threshold: {feasibility['frames_over_acceleration_review_threshold']}
- Minimum toe-body height: {feasibility['minimum_toe_body_height_m']:.3f} m
- Self-collision contacts: {feasibility['self_collision_contact_count']}
- Maximum waist speed: {feasibility['waist_speed_rad_s']['max']:.3f} rad/s
{reasons_markdown}
The acceleration review threshold is a conservative workflow diagnostic, not a
manufacturer-specified G1 actuator limit.

This is a kinematic reference-motion evaluation in MuJoCo, not a dynamically
controlled rollout and not authorization for physical-robot deployment.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--gmr-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gmr_output"))
    parser.add_argument("--robot", default="unitree_g1", choices=("unitree_g1",))
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human_frames, fps, human_height = load_human_frames(args.motion, args.model_root)
    retargeter = GeneralMotionRetargeting(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        verbose=False,
        use_velocity_limit=True,
    )
    model = retargeter.model
    data = retargeter.configuration.data
    matched_entries = {
        robot_body: values
        for robot_body, values in retargeter.ik_match_table2.items()
        if values[1] > 0
    }
    match_names = list(matched_entries)
    upper_tokens = ("shoulder", "elbow", "wrist", "torso")
    lower_tokens = ("hip", "knee", "toe")

    configurations: list[np.ndarray] = []
    position_errors: list[list[float]] = []
    feet: list[list[np.ndarray]] = []
    self_collision_contacts = 0
    task_errors: list[tuple[float, float]] = []
    started = time.perf_counter()
    for index, human_frame in enumerate(human_frames):
        configuration = retargeter.retarget(human_frame, offset_to_ground=True)
        configurations.append(configuration)
        robot_errors = []
        for robot_body, values in matched_entries.items():
            human_body = values[0]
            robot_position = data.xpos[model.body(robot_body).id]
            target_position = retargeter.scaled_human_data[human_body][0]
            robot_errors.append(float(np.linalg.norm(robot_position - target_position)))
        position_errors.append(robot_errors)
        feet.append(
            [
                data.xpos[model.body("left_toe_link").id].copy(),
                data.xpos[model.body("right_toe_link").id].copy(),
            ]
        )
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            if body1 != 0 and body2 != 0 and body1 != body2:
                self_collision_contacts += 1
        task_errors.append((float(retargeter.error1()), float(retargeter.error2())))
        if (index + 1) % 100 == 0 or index + 1 == len(human_frames):
            print(f"Retargeted {index + 1}/{len(human_frames)} frames")
    elapsed = time.perf_counter() - started

    qpos = np.asarray(configurations)
    feet_array = np.asarray(feet)
    error_array = np.asarray(position_errors)
    task_error_array = np.asarray(task_errors)
    dof_position = qpos[:, 7:]
    motor_speed = np.abs(np.diff(dof_position, axis=0)) * fps
    motor_acceleration = np.abs(np.diff(dof_position, n=2, axis=0)) * fps * fps
    acceleration_review_threshold = 50.0
    root_speed = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * fps
    root_angular_speed = quaternion_speed(qpos[:, 3:7], fps)

    joint_names = [model.joint(index).name for index in range(1, model.njnt)]
    lower_limits = np.asarray([model.jnt_range[index, 0] for index in range(1, model.njnt)])
    upper_limits = np.asarray([model.jnt_range[index, 1] for index in range(1, model.njnt)])
    limit_violations = (dof_position < lower_limits - 1e-6) | (dof_position > upper_limits + 1e-6)
    velocity_limit = 3.0 * np.pi
    velocity_violations = motor_speed > velocity_limit + 1e-6

    foot_velocity_xy = np.linalg.norm(np.diff(feet_array[:, :, :2], axis=0), axis=2) * fps
    padded_foot_velocity = np.vstack((np.zeros((1, 2)), foot_velocity_xy))
    low_height = np.percentile(feet_array[:, :, 2], 10, axis=0)
    contact = (feet_array[:, :, 2] <= low_height[None] + 0.025) & (padded_foot_velocity < 0.15)
    contact_displacements: dict[str, list[float]] = {"left": [], "right": []}
    for foot_index, side in enumerate(("left", "right")):
        for start, stop in contact_segments(contact[:, foot_index]):
            if stop - start >= 3:
                displacement = np.linalg.norm(
                    feet_array[stop - 1, foot_index, :2] - feet_array[start, foot_index, :2]
                )
                contact_displacements[side].append(float(displacement))

    upper_indices = [index for index, name in enumerate(match_names) if any(token in name for token in upper_tokens)]
    lower_indices = [index for index, name in enumerate(match_names) if any(token in name for token in lower_tokens)]
    waist_indices = [index for index, name in enumerate(joint_names) if name.startswith("waist_")]
    peak_speed_index = np.unravel_index(np.argmax(motor_speed), motor_speed.shape)
    peak_acceleration_index = np.unravel_index(
        np.argmax(motor_acceleration), motor_acceleration.shape
    )
    frames_over_acceleration_threshold = np.any(
        motor_acceleration > acceleration_review_threshold, axis=1
    )

    report: dict[str, Any] = {
        "status": "PASS",
        "scope": "kinematic simulation reference only",
        "robot": "Unitree G1 29-DoF",
        "gmr_commit": git_revision(args.gmr_repo),
        "frames": len(qpos),
        "fps_hz": fps,
        "human_height_estimate_m": human_height,
        "ground_alignment": "per-frame GMR foot offset enabled",
        "processing_seconds": elapsed,
        "processing_fps": len(qpos) / elapsed,
        "fidelity": {
            "matched_robot_bodies": match_names,
            "all_position_error_m": summary(error_array),
            "upper_body_position_error_m": summary(error_array[:, upper_indices]),
            "lower_body_position_error_m": summary(error_array[:, lower_indices]),
            "per_body_mean_error_m": {
                name: float(np.mean(error_array[:, index])) for index, name in enumerate(match_names)
            },
            "stage1_task_error": summary(task_error_array[:, 0]),
            "stage2_task_error": summary(task_error_array[:, 1]),
        },
        "feasibility": {
            "joint_limit_violation_count": int(np.sum(limit_violations)),
            "frames_with_joint_limit_violation": int(np.sum(np.any(limit_violations, axis=1))),
            "velocity_limit_rad_s": velocity_limit,
            "velocity_limit_violation_count": int(np.sum(velocity_violations)),
            "frames_with_velocity_limit_violation": int(np.sum(np.any(velocity_violations, axis=1))),
            "motor_speed_rad_s": summary(motor_speed),
            "motor_acceleration_rad_s2": summary(motor_acceleration),
            "acceleration_review_threshold_rad_s2": acceleration_review_threshold,
            "frames_over_acceleration_review_threshold": int(
                np.sum(frames_over_acceleration_threshold)
            ),
            "peak_motor_speed_event": {
                "from_frame": int(peak_speed_index[0]),
                "to_frame": int(peak_speed_index[0] + 1),
                "joint": joint_names[peak_speed_index[1]],
                "speed_rad_s": float(motor_speed[peak_speed_index]),
            },
            "peak_motor_acceleration_event": {
                "center_frame": int(peak_acceleration_index[0] + 1),
                "joint": joint_names[peak_acceleration_index[1]],
                "acceleration_rad_s2": float(
                    motor_acceleration[peak_acceleration_index]
                ),
            },
            "waist_speed_rad_s": summary(motor_speed[:, waist_indices]),
            "root_linear_speed_m_s": summary(root_speed),
            "root_angular_speed_rad_s": summary(root_angular_speed),
            "minimum_toe_body_height_m": float(np.min(feet_array[:, :, 2])),
            "self_collision_contact_count": self_collision_contacts,
            "foot_contact_frame_ratio": {
                "left": float(np.mean(contact[:, 0])),
                "right": float(np.mean(contact[:, 1])),
            },
            "foot_slip_speed_during_contact_m_s": {
                "left": summary(padded_foot_velocity[contact[:, 0], 0]) if np.any(contact[:, 0]) else None,
                "right": summary(padded_foot_velocity[contact[:, 1], 1]) if np.any(contact[:, 1]) else None,
            },
            "contact_segment_displacement_m": {
                side: summary(np.asarray(values)) if values else None
                for side, values in contact_displacements.items()
            },
            "maximum_speed_by_joint_rad_s": {
                name: float(np.max(motor_speed[:, index])) for index, name in enumerate(joint_names)
            },
        },
    }
    feasibility = report["feasibility"]
    review_reasons: list[str] = []
    if (
        feasibility["joint_limit_violation_count"] > 0
        or feasibility["velocity_limit_violation_count"] > 0
        or feasibility["minimum_toe_body_height_m"] < -0.005
        or feasibility["self_collision_contact_count"] > 0
    ):
        review_reasons.append(
            "A hard kinematic feasibility check failed (joint/velocity limit, "
            "ground penetration, or self-collision)."
        )
    if feasibility["frames_over_acceleration_review_threshold"] > 0:
        peak = feasibility["peak_motor_acceleration_event"]
        review_reasons.append(
            f"{feasibility['frames_over_acceleration_review_threshold']} frames exceed "
            f"the {acceleration_review_threshold:.0f} rad/s^2 diagnostic acceleration "
            f"threshold; the peak is {peak['acceleration_rad_s2']:.1f} rad/s^2 at "
            f"frame {peak['center_frame']} on {peak['joint']}. Smooth and re-evaluate "
            "before controller training or hardware use."
        )
    if review_reasons:
        report["status"] = "REVIEW"
        report["review_reasons"] = review_reasons

    motion_data = {
        "fps": fps,
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": dof_position,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with (args.output_dir / "ayyala_gmr_unitree_g1_29dof.pkl").open("wb") as handle:
        pickle.dump(motion_data, handle)
    np.save(args.output_dir / "ayyala_gmr_qpos_wxyz.npy", qpos)
    np.savez_compressed(
        args.output_dir / "ayyala_gmr_diagnostics.npz",
        matched_body_position_error_m=error_array,
        feet_position_m=feet_array,
        foot_contact=contact,
        motor_speed_rad_s=motor_speed,
        motor_acceleration_rad_s2=motor_acceleration,
        task_error=task_error_array,
    )
    (args.output_dir / "gmr_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "gmr_evaluation.md").write_text(markdown_report(report), encoding="utf-8")

    if not args.skip_render:
        render_motion(
            retargeter.xml_file,
            qpos,
            fps,
            args.output_dir / "ayyala_gmr_unitree_g1_29dof.mp4",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
