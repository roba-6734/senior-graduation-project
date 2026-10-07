#!/usr/bin/env python3
"""Create and validate a constrained, smoothed GMR reference motion."""

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
from qpsolvers import solve_qp
from scipy import sparse
from scipy.spatial.transform import Rotation

from general_motion_retargeting import GeneralMotionRetargeting

from run_gmr_retarget import (
    contact_segments,
    load_human_frames,
    quaternion_speed,
    render_motion,
    summary,
)


def difference_matrices(frame_count: int) -> tuple[sparse.csc_matrix, sparse.csc_matrix]:
    """Return first- and second-difference matrices for a trajectory."""
    if frame_count < 3:
        raise ValueError("At least three frames are required for trajectory smoothing")
    first = sparse.diags(
        (-np.ones(frame_count - 1), np.ones(frame_count - 1)),
        (0, 1),
        shape=(frame_count - 1, frame_count),
        format="csc",
    )
    second = sparse.diags(
        (np.ones(frame_count - 2), -2.0 * np.ones(frame_count - 2), np.ones(frame_count - 2)),
        (0, 1, 2),
        shape=(frame_count - 2, frame_count),
        format="csc",
    )
    return first, second


def smooth_joint_trajectories(
    raw_dof_position: np.ndarray,
    fps: float,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    velocity_limit_rad_s: float,
    acceleration_limit_rad_s2: float,
    smoothness_weight: float,
) -> np.ndarray:
    """Find the closest smooth trajectory satisfying hard kinematic bounds.

    Each joint is an independent convex quadratic program. The objective keeps
    the result close to GMR while penalizing second differences. Joint ranges,
    frame-to-frame velocity, and frame-to-frame acceleration are constraints.
    """
    raw = np.asarray(raw_dof_position, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D joint trajectory, got {raw.shape}")
    frame_count, joint_count = raw.shape
    if lower_limits.shape != (joint_count,) or upper_limits.shape != (joint_count,):
        raise ValueError("Joint-limit arrays do not match the trajectory")
    if fps <= 0 or velocity_limit_rad_s <= 0 or acceleration_limit_rad_s2 <= 0:
        raise ValueError("FPS and trajectory limits must be positive")
    if smoothness_weight < 0:
        raise ValueError("Smoothness weight cannot be negative")

    first, second = difference_matrices(frame_count)
    identity = sparse.eye(frame_count, format="csc")
    objective = 2.0 * (
        identity + smoothness_weight * sparse.csc_matrix(second.T @ second)
    )
    constraints = sparse.vstack((first, -first, second, -second), format="csc")
    maximum_step = velocity_limit_rad_s / fps
    maximum_second_difference = acceleration_limit_rad_s2 / (fps * fps)
    constraint_bounds = np.concatenate(
        (
            np.full(frame_count - 1, maximum_step),
            np.full(frame_count - 1, maximum_step),
            np.full(frame_count - 2, maximum_second_difference),
            np.full(frame_count - 2, maximum_second_difference),
        )
    )

    result = np.empty_like(raw)
    for joint_index in range(joint_count):
        solution = solve_qp(
            objective,
            -2.0 * raw[:, joint_index],
            constraints,
            constraint_bounds,
            lb=np.full(frame_count, lower_limits[joint_index]),
            ub=np.full(frame_count, upper_limits[joint_index]),
            solver="proxqp",
        )
        if solution is None or not np.all(np.isfinite(solution)):
            raise RuntimeError(f"Smoothing QP failed for joint index {joint_index}")
        result[:, joint_index] = solution
    return result


def qpos_to_optimization_state(qpos: np.ndarray) -> np.ndarray:
    """Convert MuJoCo qpos into xyz, continuous rotation-vector, and joint state."""
    root_rotation = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_rotvec()
    return np.column_stack((qpos[:, :3], root_rotation, qpos[:, 7:]))


def optimization_state_to_qpos(
    state: np.ndarray, reference_qpos: np.ndarray
) -> np.ndarray:
    """Convert the smoothing state back to MuJoCo WXYZ qpos."""
    qpos = reference_qpos.copy()
    qpos[:, :3] = state[:, :3]
    quaternion_xyzw = Rotation.from_rotvec(state[:, 3:6]).as_quat()
    qpos[:, 3:7] = quaternion_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = state[:, 6:]
    return qpos


def event_windows(
    raw_state: np.ndarray,
    fps: float,
    motor_acceleration_limit: float,
    root_linear_acceleration_limit: float,
    root_angular_acceleration_limit: float,
    padding: int,
) -> list[tuple[int, int]]:
    """Find and merge padded windows containing raw acceleration discontinuities."""
    acceleration = np.abs(np.diff(raw_state, n=2, axis=0)) * fps * fps
    event_mask = (
        np.any(acceleration[:, :3] > root_linear_acceleration_limit, axis=1)
        | np.any(acceleration[:, 3:6] > root_angular_acceleration_limit, axis=1)
        | np.any(acceleration[:, 6:] > motor_acceleration_limit, axis=1)
    )
    centers = np.flatnonzero(event_mask) + 1
    windows: list[tuple[int, int]] = []
    for center in centers:
        start = max(0, int(center) - padding)
        stop = min(len(raw_state), int(center) + padding + 1)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(stop, windows[-1][1]))
        else:
            windows.append((start, stop))
    return windows


def foot_positions_and_state_jacobians(
    model: mujoco.MjModel,
    reference_qpos: np.ndarray,
    state: np.ndarray,
    rotation_epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute toe positions and Jacobians with respect to the 35-state vector."""
    data = mujoco.MjData(model)
    qpos = optimization_state_to_qpos(state, reference_qpos)
    foot_names = ("left_toe_link", "right_toe_link")
    foot_ids = [model.body(name).id for name in foot_names]
    positions = np.empty((len(state), 2, 3), dtype=np.float64)
    jacobians = np.empty((len(state), 2, 3, 35), dtype=np.float64)

    for frame_index, configuration in enumerate(qpos):
        data.qpos[:] = configuration
        mujoco.mj_forward(model, data)
        for foot_index, body_id in enumerate(foot_ids):
            positions[frame_index, foot_index] = data.xpos[body_id]
            jacobian_position = np.zeros((3, model.nv))
            jacobian_rotation = np.zeros((3, model.nv))
            mujoco.mj_jacBody(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                body_id,
            )
            jacobians[frame_index, foot_index, :, :3] = jacobian_position[:, :3]
            jacobians[frame_index, foot_index, :, 6:] = jacobian_position[:, 6:]

        # MuJoCo exposes an angular-tangent Jacobian, while the optimizer uses
        # absolute rotation vectors. A centered finite difference supplies the
        # correct local derivative for those three state coordinates.
        for rotation_axis in range(3):
            plus_state = state[frame_index].copy()
            minus_state = state[frame_index].copy()
            plus_state[3 + rotation_axis] += rotation_epsilon
            minus_state[3 + rotation_axis] -= rotation_epsilon
            perturbed = optimization_state_to_qpos(
                np.vstack((plus_state, minus_state)),
                np.vstack((reference_qpos[frame_index], reference_qpos[frame_index])),
            )
            perturbed_positions = []
            for perturbed_qpos in perturbed:
                data.qpos[:] = perturbed_qpos
                mujoco.mj_forward(model, data)
                perturbed_positions.append(
                    np.asarray([data.xpos[body_id].copy() for body_id in foot_ids])
                )
            derivative = (
                perturbed_positions[0] - perturbed_positions[1]
            ) / (2.0 * rotation_epsilon)
            jacobians[frame_index, :, :, 3 + rotation_axis] = derivative

    return positions, jacobians


def smooth_reference_motion(
    raw_qpos: np.ndarray,
    contact: np.ndarray,
    model: mujoco.MjModel,
    fps: float,
    motor_acceleration_limit: float,
    smoothness_weight: float,
    iterations: int = 5,
    window_padding: int = 8,
    root_linear_velocity_limit: float = 0.5,
    root_linear_acceleration_limit: float = 3.0,
    root_angular_velocity_limit: float = 3.0,
    root_angular_acceleration_limit: float = 25.0,
    foot_position_weight: float = 1_000.0,
    foot_step_tolerance_m: float = 0.00025,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Smooth event windows with contact-aware sequential convex optimization."""
    if iterations < 1 or window_padding < 2:
        raise ValueError("At least one iteration and two padding frames are required")
    raw_state = qpos_to_optimization_state(raw_qpos)
    state = raw_state.copy()
    state_size = raw_state.shape[1]
    windows = event_windows(
        raw_state,
        fps,
        motor_acceleration_limit,
        root_linear_acceleration_limit,
        root_angular_acceleration_limit,
        window_padding,
    )
    raw_feet, _ = foot_positions_and_state_jacobians(
        model, raw_qpos, raw_state
    )
    lower_joint_limits = np.asarray(
        [model.jnt_range[index, 0] for index in range(1, model.njnt)]
    )
    upper_joint_limits = np.asarray(
        [model.jnt_range[index, 1] for index in range(1, model.njnt)]
    )
    velocity_limits = np.concatenate(
        (
            np.full(3, root_linear_velocity_limit),
            np.full(3, root_angular_velocity_limit),
            np.full(29, 3.0 * np.pi),
        )
    )
    acceleration_limits = np.concatenate(
        (
            np.full(3, root_linear_acceleration_limit),
            np.full(3, root_angular_acceleration_limit),
            np.full(29, motor_acceleration_limit),
        )
    )
    fidelity_weights = np.concatenate(
        (np.full(3, 100.0), np.full(3, 10.0), np.ones(29))
    )
    smoothness_weights = np.concatenate(
        (np.full(3, 10.0), np.ones(3), np.ones(29))
    )

    for start, stop in windows:
        window_state = state[start:stop].copy()
        raw_window_state = raw_state[start:stop]
        reference_window_qpos = raw_qpos[start:stop]
        frame_count = len(window_state)
        first, second = difference_matrices(frame_count)
        first_all = sparse.kron(
            first, sparse.eye(state_size), format="csc"
        )
        second_all = sparse.kron(
            second, sparse.eye(state_size), format="csc"
        )
        fidelity_matrix = sparse.diags(
            np.tile(fidelity_weights, frame_count), format="csc"
        )
        smoothness_matrix = sparse.diags(
            np.tile(smoothness_weights * smoothness_weight, frame_count - 2),
            format="csc",
        )
        smoothness_quadratic = (
            second_all.T @ smoothness_matrix @ second_all
        )

        for _ in range(iterations):
            current_feet, jacobians = foot_positions_and_state_jacobians(
                model, reference_window_qpos, window_state
            )
            foot_blocks = []
            foot_linear_terms = []
            for local_index in range(frame_count):
                global_index = start + local_index
                active_jacobians = []
                position_residuals = []
                for foot_index in range(2):
                    if contact[global_index, foot_index]:
                        active_jacobians.append(jacobians[local_index, foot_index])
                        position_residuals.append(
                            current_feet[local_index, foot_index]
                            - raw_feet[global_index, foot_index]
                        )
                if active_jacobians:
                    active = np.vstack(active_jacobians)
                    residual = np.concatenate(position_residuals)
                    foot_blocks.append(sparse.csc_matrix(active.T @ active))
                    foot_linear_terms.append(active.T @ residual)
                else:
                    foot_blocks.append(
                        sparse.csc_matrix((state_size, state_size))
                    )
                    foot_linear_terms.append(np.zeros(state_size))
            foot_quadratic = sparse.block_diag(foot_blocks, format="csc")
            foot_linear = np.concatenate(foot_linear_terms)
            flat_state = window_state.reshape(-1)
            flat_raw_state = raw_window_state.reshape(-1)
            objective = 2.0 * (
                fidelity_matrix
                + smoothness_quadratic
                + foot_position_weight * foot_quadratic
            )
            linear = 2.0 * (
                fidelity_matrix @ (flat_state - flat_raw_state)
                + smoothness_quadratic @ flat_state
                + foot_position_weight * foot_linear
            )

            foot_step_rows = []
            foot_step_residuals = []
            for local_index in range(1, frame_count):
                global_index = start + local_index
                for foot_index in range(2):
                    # The slip metric is assigned to the current frame, so a
                    # contact-entry transition must be constrained as well.
                    if not contact[global_index, foot_index]:
                        continue
                    correction_step = (
                        current_feet[local_index, foot_index, :2]
                        - current_feet[local_index - 1, foot_index, :2]
                        - raw_feet[global_index, foot_index, :2]
                        + raw_feet[global_index - 1, foot_index, :2]
                    )
                    for axis in range(2):
                        row = sparse.lil_matrix(
                            (1, frame_count * state_size)
                        )
                        previous_slice = slice(
                            (local_index - 1) * state_size,
                            local_index * state_size,
                        )
                        current_slice = slice(
                            local_index * state_size,
                            (local_index + 1) * state_size,
                        )
                        row[0, previous_slice] = -jacobians[
                            local_index - 1, foot_index, axis
                        ]
                        row[0, current_slice] = jacobians[
                            local_index, foot_index, axis
                        ]
                        foot_step_rows.append(row.tocsc())
                        foot_step_residuals.append(correction_step[axis])
            foot_step_matrix = (
                sparse.vstack(foot_step_rows, format="csc")
                if foot_step_rows
                else sparse.csc_matrix((0, frame_count * state_size))
            )
            foot_step_residual = np.asarray(foot_step_residuals)

            constraints = sparse.vstack(
                (
                    first_all,
                    -first_all,
                    second_all,
                    -second_all,
                    foot_step_matrix,
                    -foot_step_matrix,
                ),
                format="csc",
            )
            first_value = first_all @ flat_state
            second_value = second_all @ flat_state
            maximum_step = np.tile(
                velocity_limits / fps, frame_count - 1
            )
            maximum_second_difference = np.tile(
                acceleration_limits / (fps * fps), frame_count - 2
            )
            constraint_bounds = np.concatenate(
                (
                    maximum_step - first_value,
                    maximum_step + first_value,
                    maximum_second_difference - second_value,
                    maximum_second_difference + second_value,
                    np.full(len(foot_step_residual), foot_step_tolerance_m)
                    - foot_step_residual,
                    np.full(len(foot_step_residual), foot_step_tolerance_m)
                    + foot_step_residual,
                )
            )
            state_lower = np.concatenate(
                (np.full(6, -np.inf), lower_joint_limits)
            )
            state_upper = np.concatenate(
                (np.full(6, np.inf), upper_joint_limits)
            )
            lower = np.tile(state_lower, frame_count) - flat_state
            upper = np.tile(state_upper, frame_count) - flat_state
            # Two untouched padding frames make velocity and acceleration
            # continuous at each interior window boundary.
            if start > 0:
                lower[: 2 * state_size] = 0.0
                upper[: 2 * state_size] = 0.0
            if stop < len(state):
                lower[-2 * state_size :] = 0.0
                upper[-2 * state_size :] = 0.0

            delta = solve_qp(
                objective,
                linear,
                constraints,
                constraint_bounds,
                lb=lower,
                ub=upper,
                solver="proxqp",
            )
            if delta is None or not np.all(np.isfinite(delta)):
                raise RuntimeError(
                    f"Contact-aware smoothing failed for window {start}:{stop}"
                )
            window_state += delta.reshape(frame_count, state_size)
        state[start:stop] = window_state

    metadata = {
        "event_windows": [[start, stop] for start, stop in windows],
        "optimized_frame_count": int(sum(stop - start for start, stop in windows)),
        "sequential_convex_iterations": iterations,
        "window_padding_frames": window_padding,
        "root_linear_velocity_limit_m_s": root_linear_velocity_limit,
        "root_linear_acceleration_limit_m_s2_per_axis": root_linear_acceleration_limit,
        "root_angular_velocity_limit_rad_s": root_angular_velocity_limit,
        "root_angular_acceleration_limit_rad_s2_per_axis": root_angular_acceleration_limit,
        "foot_position_weight": foot_position_weight,
        "foot_step_tolerance_m_per_axis": foot_step_tolerance_m,
    }
    return optimization_state_to_qpos(state, raw_qpos), metadata


def foot_motion_metrics(
    feet: np.ndarray,
    fps: float,
    contact: np.ndarray,
) -> tuple[dict[str, dict[str, float] | None], dict[str, dict[str, float] | None]]:
    """Measure foot speed and displacement using a supplied contact mask."""
    velocity_xy = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) * fps
    padded_velocity = np.vstack((np.zeros((1, 2)), velocity_xy))
    slip_speed: dict[str, dict[str, float] | None] = {}
    segment_displacement: dict[str, dict[str, float] | None] = {}
    for foot_index, side in enumerate(("left", "right")):
        mask = contact[:, foot_index]
        slip_speed[side] = summary(padded_velocity[mask, foot_index]) if np.any(mask) else None
        displacements = []
        for start, stop in contact_segments(mask):
            if stop - start >= 3:
                displacements.append(
                    float(
                        np.linalg.norm(
                            feet[stop - 1, foot_index, :2]
                            - feet[start, foot_index, :2]
                        )
                    )
                )
        segment_displacement[side] = (
            summary(np.asarray(displacements)) if displacements else None
        )
    return slip_speed, segment_displacement


def evaluate_motion(
    qpos: np.ndarray,
    human_frames: list[dict],
    retargeter: GeneralMotionRetargeting,
    fps: float,
    reference_contact: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate fidelity and kinematic feasibility from fresh MuJoCo FK."""
    model = retargeter.model
    data = retargeter.configuration.data
    matched_entries = {
        robot_body: values
        for robot_body, values in retargeter.ik_match_table2.items()
        if values[1] > 0
    }
    match_names = list(matched_entries)
    errors: list[list[float]] = []
    feet: list[list[np.ndarray]] = []
    task_errors: list[tuple[float, float]] = []
    self_collision_contacts = 0
    minimum_ground_contact_distance = np.inf

    for configuration, human_frame in zip(qpos, human_frames, strict=True):
        retargeter.update_targets(human_frame, offset_to_ground=True)
        data.qpos[:] = configuration
        mujoco.mj_forward(model, data)
        errors.append(
            [
                float(
                    np.linalg.norm(
                        data.xpos[model.body(robot_body).id]
                        - retargeter.scaled_human_data[values[0]][0]
                    )
                )
                for robot_body, values in matched_entries.items()
            ]
        )
        feet.append(
            [
                data.xpos[model.body("left_toe_link").id].copy(),
                data.xpos[model.body("right_toe_link").id].copy(),
            ]
        )
        task_errors.append((float(retargeter.error1()), float(retargeter.error2())))
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            if body1 == 0 or body2 == 0:
                minimum_ground_contact_distance = min(
                    minimum_ground_contact_distance, float(contact.dist)
                )
            elif body1 != body2:
                self_collision_contacts += 1

    error_array = np.asarray(errors)
    feet_array = np.asarray(feet)
    task_error_array = np.asarray(task_errors)
    dof_position = qpos[:, 7:]
    speed = np.abs(np.diff(dof_position, axis=0)) * fps
    acceleration = np.abs(np.diff(dof_position, n=2, axis=0)) * fps * fps
    root_speed = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * fps
    root_acceleration = (
        np.linalg.norm(np.diff(qpos[:, :3], n=2, axis=0), axis=1) * fps * fps
    )
    root_angular_speed = quaternion_speed(qpos[:, 3:7], fps)
    root_rotation = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    root_angular_velocity = (
        root_rotation[:-1].inv() * root_rotation[1:]
    ).as_rotvec() * fps
    root_angular_acceleration = (
        np.linalg.norm(np.diff(root_angular_velocity, axis=0), axis=1) * fps
    )

    joint_names = [model.joint(index).name for index in range(1, model.njnt)]
    lower_limits = np.asarray(
        [model.jnt_range[index, 0] for index in range(1, model.njnt)]
    )
    upper_limits = np.asarray(
        [model.jnt_range[index, 1] for index in range(1, model.njnt)]
    )
    limit_violations = (dof_position < lower_limits - 1e-6) | (
        dof_position > upper_limits + 1e-6
    )
    velocity_limit = 3.0 * np.pi
    velocity_violations = speed > velocity_limit + 1e-6

    foot_velocity_xy = np.linalg.norm(np.diff(feet_array[:, :, :2], axis=0), axis=2) * fps
    padded_foot_velocity = np.vstack((np.zeros((1, 2)), foot_velocity_xy))
    low_height = np.percentile(feet_array[:, :, 2], 10, axis=0)
    inferred_contact = (feet_array[:, :, 2] <= low_height[None] + 0.025) & (
        padded_foot_velocity < 0.15
    )
    measurement_contact = inferred_contact if reference_contact is None else reference_contact
    slip_speed, segment_displacement = foot_motion_metrics(
        feet_array, fps, measurement_contact
    )

    upper_tokens = ("shoulder", "elbow", "wrist", "torso")
    lower_tokens = ("hip", "knee", "toe")
    upper_indices = [
        index
        for index, name in enumerate(match_names)
        if any(token in name for token in upper_tokens)
    ]
    lower_indices = [
        index
        for index, name in enumerate(match_names)
        if any(token in name for token in lower_tokens)
    ]
    waist_indices = [
        index for index, name in enumerate(joint_names) if name.startswith("waist_")
    ]
    peak_speed_index = np.unravel_index(np.argmax(speed), speed.shape)
    peak_acceleration_index = np.unravel_index(np.argmax(acceleration), acceleration.shape)

    metrics: dict[str, Any] = {
        "fidelity": {
            "all_position_error_m": summary(error_array),
            "upper_body_position_error_m": summary(error_array[:, upper_indices]),
            "lower_body_position_error_m": summary(error_array[:, lower_indices]),
            "per_body_mean_error_m": {
                name: float(np.mean(error_array[:, index]))
                for index, name in enumerate(match_names)
            },
            "stage1_task_error": summary(task_error_array[:, 0]),
            "stage2_task_error": summary(task_error_array[:, 1]),
        },
        "feasibility": {
            "joint_limit_violation_count": int(np.sum(limit_violations)),
            "frames_with_joint_limit_violation": int(
                np.sum(np.any(limit_violations, axis=1))
            ),
            "velocity_limit_rad_s": velocity_limit,
            "velocity_limit_violation_count": int(np.sum(velocity_violations)),
            "frames_with_velocity_limit_violation": int(
                np.sum(np.any(velocity_violations, axis=1))
            ),
            "motor_speed_rad_s": summary(speed),
            "motor_acceleration_rad_s2": summary(acceleration),
            "peak_motor_speed_event": {
                "from_frame": int(peak_speed_index[0]),
                "to_frame": int(peak_speed_index[0] + 1),
                "joint": joint_names[peak_speed_index[1]],
                "speed_rad_s": float(speed[peak_speed_index]),
            },
            "peak_motor_acceleration_event": {
                "center_frame": int(peak_acceleration_index[0] + 1),
                "joint": joint_names[peak_acceleration_index[1]],
                "acceleration_rad_s2": float(acceleration[peak_acceleration_index]),
            },
            "waist_speed_rad_s": summary(speed[:, waist_indices]),
            "root_linear_speed_m_s": summary(root_speed),
            "root_linear_acceleration_m_s2": summary(root_acceleration),
            "root_angular_speed_rad_s": summary(root_angular_speed),
            "root_angular_acceleration_rad_s2": summary(
                root_angular_acceleration
            ),
            "minimum_toe_body_height_m": float(np.min(feet_array[:, :, 2])),
            "minimum_ground_contact_distance_m": (
                float(minimum_ground_contact_distance)
                if np.isfinite(minimum_ground_contact_distance)
                else None
            ),
            "self_collision_contact_count": self_collision_contacts,
            "inferred_foot_contact_frame_ratio": {
                "left": float(np.mean(inferred_contact[:, 0])),
                "right": float(np.mean(inferred_contact[:, 1])),
            },
            "foot_slip_speed_during_reference_contact_m_s": slip_speed,
            "contact_segment_displacement_during_reference_contact_m": segment_displacement,
            "maximum_speed_by_joint_rad_s": {
                name: float(np.max(speed[:, index]))
                for index, name in enumerate(joint_names)
            },
        },
    }
    arrays = {
        "matched_body_position_error_m": error_array,
        "feet_position_m": feet_array,
        "inferred_foot_contact": inferred_contact,
        "motor_speed_rad_s": speed,
        "motor_acceleration_rad_s2": acceleration,
        "task_error": task_error_array,
    }
    return metrics, arrays


def maximum_segment_displacement(metrics: dict[str, Any]) -> float:
    values = metrics["feasibility"][
        "contact_segment_displacement_during_reference_contact_m"
    ]
    maxima = [entry["max"] for entry in values.values() if entry is not None]
    return max(maxima, default=0.0)


def render_comparison(raw_video: Path, smoothed_video: Path, output_path: Path) -> None:
    """Create a side-by-side H.264 comparison with v1 on the left and v2 on the right."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_video),
            "-i",
            str(smoothed_video),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )


def make_contact_sheet(video_path: Path, output_path: Path, frames: int = 16) -> None:
    """Save a 4x4 visual summary from evenly spaced video frames."""
    reader = imageio.get_reader(video_path)
    try:
        frame_count = int(reader.count_frames())
        indices = np.linspace(0, frame_count - 1, frames, dtype=int)
        images = [reader.get_data(int(index)) for index in indices]
    finally:
        reader.close()
    rows = []
    for row_start in range(0, frames, 4):
        rows.append(np.hstack(images[row_start : row_start + 4]))
    imageio.imwrite(output_path, np.vstack(rows), quality=88)


def markdown_report(report: dict[str, Any]) -> str:
    raw = report["raw"]
    smooth = report["smoothed"]
    checks = report["acceptance_checks"]
    comparison = report["comparison"]
    check_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}"
        for name, passed in checks.items()
    )
    return f"""# Smoothed GMR v2 evaluation

**Status:** {report['status']}

| Metric | Raw v1 | Smoothed v2 |
|---|---:|---:|
| Mean matched-body error | {raw['fidelity']['all_position_error_m']['mean'] * 100:.2f} cm | {smooth['fidelity']['all_position_error_m']['mean'] * 100:.2f} cm |
| Mean upper-body error | {raw['fidelity']['upper_body_position_error_m']['mean'] * 100:.2f} cm | {smooth['fidelity']['upper_body_position_error_m']['mean'] * 100:.2f} cm |
| Maximum motor speed | {raw['feasibility']['motor_speed_rad_s']['max']:.2f} rad/s | {smooth['feasibility']['motor_speed_rad_s']['max']:.2f} rad/s |
| P99 motor acceleration | {raw['feasibility']['motor_acceleration_rad_s2']['p99']:.2f} rad/s^2 | {smooth['feasibility']['motor_acceleration_rad_s2']['p99']:.2f} rad/s^2 |
| Maximum motor acceleration | {raw['feasibility']['motor_acceleration_rad_s2']['max']:.2f} rad/s^2 | {smooth['feasibility']['motor_acceleration_rad_s2']['max']:.2f} rad/s^2 |
| Maximum root linear acceleration | {raw['feasibility']['root_linear_acceleration_m_s2']['max']:.2f} m/s^2 | {smooth['feasibility']['root_linear_acceleration_m_s2']['max']:.2f} m/s^2 |
| Maximum root angular acceleration | {raw['feasibility']['root_angular_acceleration_rad_s2']['max']:.2f} rad/s^2 | {smooth['feasibility']['root_angular_acceleration_rad_s2']['max']:.2f} rad/s^2 |
| Maximum contact-segment displacement | {comparison['raw_max_contact_segment_displacement_m'] * 100:.2f} cm | {comparison['smoothed_max_contact_segment_displacement_m'] * 100:.2f} cm |

Mean absolute joint adjustment: {comparison['joint_adjustment_rad']['mean']:.5f} rad.

## Acceptance checks

{check_lines}

The 40 rad/s^2 smoothing constraint and 50 rad/s^2 acceptance threshold are
conservative project diagnostics, not manufacturer-specified actuator limits.
This is still a kinematic reference. PASS means it is ready for dynamic
tracking tests in simulation, not deployment on a physical robot.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-qpos", type=Path, required=True)
    parser.add_argument("--raw-video", type=Path)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gmr_output_v2"))
    parser.add_argument("--acceleration-limit", type=float, default=40.0)
    parser.add_argument("--acceptance-acceleration", type=float, default=50.0)
    parser.add_argument("--smoothness-weight", type=float, default=1.0)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_qpos = np.load(args.raw_qpos)
    human_frames, fps, human_height = load_human_frames(args.motion, args.model_root)
    if raw_qpos.shape != (len(human_frames), 36):
        raise ValueError(
            f"Expected {(len(human_frames), 36)} qpos values, got {raw_qpos.shape}"
        )

    retargeter = GeneralMotionRetargeting(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
        use_velocity_limit=True,
    )
    model = retargeter.model

    print("Evaluating raw v1 reference with fresh forward kinematics")
    raw_metrics, raw_arrays = evaluate_motion(
        raw_qpos, human_frames, retargeter, fps
    )
    started = time.perf_counter()
    smoothed_qpos, smoothing_metadata = smooth_reference_motion(
        raw_qpos,
        raw_arrays["inferred_foot_contact"],
        model,
        fps,
        motor_acceleration_limit=args.acceleration_limit,
        smoothness_weight=args.smoothness_weight,
    )
    smoothing_seconds = time.perf_counter() - started

    print("Evaluating smoothed v2 reference against v1 contacts")
    smooth_metrics, smooth_arrays = evaluate_motion(
        smoothed_qpos,
        human_frames,
        retargeter,
        fps,
        reference_contact=raw_arrays["inferred_foot_contact"],
    )

    adjustment = np.abs(smoothed_qpos[:, 7:] - raw_qpos[:, 7:])
    root_position_adjustment = np.linalg.norm(
        smoothed_qpos[:, :3] - raw_qpos[:, :3], axis=1
    )
    raw_root_rotation = Rotation.from_quat(raw_qpos[:, [4, 5, 6, 3]])
    smooth_root_rotation = Rotation.from_quat(smoothed_qpos[:, [4, 5, 6, 3]])
    root_rotation_adjustment = (
        raw_root_rotation.inv() * smooth_root_rotation
    ).magnitude()
    raw_segment_max = maximum_segment_displacement(raw_metrics)
    smooth_segment_max = maximum_segment_displacement(smooth_metrics)
    smooth_fidelity = smooth_metrics["fidelity"]
    smooth_feasibility = smooth_metrics["feasibility"]
    raw_feasibility = raw_metrics["feasibility"]
    raw_slip = raw_feasibility["foot_slip_speed_during_reference_contact_m_s"]
    smooth_slip = smooth_feasibility[
        "foot_slip_speed_during_reference_contact_m_s"
    ]
    slip_mean_ok = all(
        smooth_slip[side]["mean"] <= raw_slip[side]["mean"] + 0.002
        for side in ("left", "right")
    )
    slip_p95_ok = all(
        smooth_slip[side]["p95"] <= raw_slip[side]["p95"] + 0.01
        for side in ("left", "right")
    )
    slip_max_ok = all(
        smooth_slip[side]["max"] <= raw_slip[side]["max"] + 0.02
        for side in ("left", "right")
    )
    acceptance_checks = {
        "all_623_frames_preserved": len(smoothed_qpos) == 623,
        "zero_joint_limit_violations": smooth_feasibility[
            "joint_limit_violation_count"
        ]
        == 0,
        "zero_velocity_limit_violations": smooth_feasibility[
            "velocity_limit_violation_count"
        ]
        == 0,
        "maximum_acceleration_below_50_rad_s2": smooth_feasibility[
            "motor_acceleration_rad_s2"
        ]["max"]
        <= args.acceptance_acceleration + 1e-4,
        "root_linear_acceleration_below_6_m_s2": smooth_feasibility[
            "root_linear_acceleration_m_s2"
        ]["max"]
        <= 6.0,
        "root_angular_acceleration_below_50_rad_s2": smooth_feasibility[
            "root_angular_acceleration_rad_s2"
        ]["max"]
        <= 50.0,
        "mean_position_error_at_most_8_5_cm": smooth_fidelity[
            "all_position_error_m"
        ]["mean"]
        <= 0.085,
        "mean_error_increase_below_1_cm": smooth_fidelity[
            "all_position_error_m"
        ]["mean"]
        <= raw_metrics["fidelity"]["all_position_error_m"]["mean"] + 0.01,
        "mean_upper_body_error_at_most_10_cm": smooth_fidelity[
            "upper_body_position_error_m"
        ]["mean"]
        <= 0.10,
        "zero_self_collision_contacts": smooth_feasibility[
            "self_collision_contact_count"
        ]
        == 0,
        "no_toe_body_ground_penetration": smooth_feasibility[
            "minimum_toe_body_height_m"
        ]
        >= -0.005,
        "contact_displacement_not_increased_over_5_mm": smooth_segment_max
        <= raw_segment_max + 0.005,
        "mean_contact_foot_speed_not_increased_over_2_mm_s": slip_mean_ok,
        "p95_contact_foot_speed_not_increased_over_1_cm_s": slip_p95_ok,
        "maximum_contact_foot_speed_not_increased_over_2_cm_s": slip_max_ok,
    }
    status = "PASS" if all(acceptance_checks.values()) else "REVIEW"
    report: dict[str, Any] = {
        "status": status,
        "scope": "kinematic reference; ready only for dynamic simulation when PASS",
        "robot": "Unitree G1 29-DoF",
        "frames": len(smoothed_qpos),
        "fps_hz": fps,
        "smoothing": {
            "method": "contact-aware sequential convex optimization of event windows",
            "solver": "proxqp",
            "smoothness_weight": args.smoothness_weight,
            "hard_motor_velocity_limit_rad_s": 3.0 * np.pi,
            "hard_motor_acceleration_limit_rad_s2": args.acceleration_limit,
            "processing_seconds": smoothing_seconds,
            **smoothing_metadata,
        },
        "comparison": {
            "joint_adjustment_rad": summary(adjustment),
            "root_position_adjustment_m": summary(root_position_adjustment),
            "root_rotation_adjustment_rad": summary(root_rotation_adjustment),
            "raw_max_contact_segment_displacement_m": raw_segment_max,
            "smoothed_max_contact_segment_displacement_m": smooth_segment_max,
        },
        "acceptance_checks": acceptance_checks,
        "raw": raw_metrics,
        "smoothed": smooth_metrics,
    }

    motion_data = {
        "fps": fps,
        "root_pos": smoothed_qpos[:, :3],
        "root_rot": smoothed_qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": smoothed_qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    pickle_path = args.output_dir / "ayyala_gmr_unitree_g1_29dof_v2_smoothed.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(motion_data, handle)
    np.save(
        args.output_dir / "ayyala_gmr_unitree_g1_29dof_v2_smoothed_qpos_wxyz.npy",
        smoothed_qpos,
    )
    np.savez_compressed(
        args.output_dir / "ayyala_gmr_v2_diagnostics.npz",
        raw_matched_body_position_error_m=raw_arrays[
            "matched_body_position_error_m"
        ],
        smoothed_matched_body_position_error_m=smooth_arrays[
            "matched_body_position_error_m"
        ],
        raw_feet_position_m=raw_arrays["feet_position_m"],
        smoothed_feet_position_m=smooth_arrays["feet_position_m"],
        reference_foot_contact=raw_arrays["inferred_foot_contact"],
        smoothed_inferred_foot_contact=smooth_arrays["inferred_foot_contact"],
        raw_motor_speed_rad_s=raw_arrays["motor_speed_rad_s"],
        smoothed_motor_speed_rad_s=smooth_arrays["motor_speed_rad_s"],
        raw_motor_acceleration_rad_s2=raw_arrays[
            "motor_acceleration_rad_s2"
        ],
        smoothed_motor_acceleration_rad_s2=smooth_arrays[
            "motor_acceleration_rad_s2"
        ],
        joint_adjustment_rad=smoothed_qpos[:, 7:] - raw_qpos[:, 7:],
        root_position_adjustment_m=smoothed_qpos[:, :3] - raw_qpos[:, :3],
        root_rotation_adjustment_rad=root_rotation_adjustment,
    )
    report_path = args.output_dir / "gmr_v2_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "gmr_v2_evaluation.md").write_text(
        markdown_report(report), encoding="utf-8"
    )

    if not args.skip_render:
        smooth_video = (
            args.output_dir / "ayyala_gmr_unitree_g1_29dof_v2_smoothed.mp4"
        )
        print("Rendering smoothed v2 preview")
        render_motion(retargeter.xml_file, smoothed_qpos, fps, smooth_video)
        make_contact_sheet(
            smooth_video, args.output_dir / "gmr_v2_contact_sheet.jpg"
        )
        if args.raw_video is not None and args.raw_video.exists():
            comparison_video = args.output_dir / "ayyala_gmr_v1_vs_v2.mp4"
            render_comparison(args.raw_video, smooth_video, comparison_video)
            make_contact_sheet(
                comparison_video,
                args.output_dir / "gmr_v1_vs_v2_contact_sheet.jpg",
            )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
