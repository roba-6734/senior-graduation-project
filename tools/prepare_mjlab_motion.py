#!/usr/bin/env python3
"""Export and validate Unitree RL MjLab motion files.

The official Unitree CSV schema is:

  root_xyz (3), root_quaternion_xyzw (4), joint_position (29)

The CSV intentionally has no header because ``scripts/csv_to_npz.py`` uses
``numpy.loadtxt``.  Pickle input must be trusted; Python pickle is not a safe
format for untrusted files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


UNITREE_RL_MJLAB_COMMIT = "1425b15f73bd4095f0df53709d7c389c3eb9e790"

G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

NPZ_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require_array(
    payload: dict[str, Any], key: str, shape_tail: tuple[int, ...]
) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"Missing required pickle field: {key}")
    array = np.asarray(payload[key], dtype=np.float64)
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise ValueError(
            f"{key} must have shape (frames, {', '.join(map(str, shape_tail))}); "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains NaN or infinity")
    return array


def scalar_fps(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"fps must contain one value; got shape {np.asarray(value).shape}")
    fps = float(array[0])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive; got {fps}")
    return fps


def export_pickle(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else Path(str(output_path) + ".manifest.json")
    )

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    with input_path.open("rb") as stream:
        payload = pickle.load(stream)  # noqa: S301 - project-generated trusted input.
    if not isinstance(payload, dict):
        raise ValueError(f"Expected pickle dictionary, got {type(payload).__name__}")

    root_pos = require_array(payload, "root_pos", (3,))
    root_rot = require_array(payload, "root_rot", (4,))
    dof_pos = require_array(payload, "dof_pos", (len(G1_29DOF_JOINT_NAMES),))
    frame_count = root_pos.shape[0]
    if frame_count < 2:
        raise ValueError(f"At least two frames are required; got {frame_count}")
    if root_rot.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
        raise ValueError(
            "root_pos, root_rot, and dof_pos must have the same frame count"
        )

    fps = scalar_fps(payload.get("fps"))
    if args.expected_fps is not None and not math.isclose(
        fps, args.expected_fps, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError(f"Expected {args.expected_fps:g} FPS, but pickle has {fps:g}")

    quat_norm = np.linalg.norm(root_rot, axis=1)
    max_norm_error = float(np.max(np.abs(quat_norm - 1.0)))
    if max_norm_error > args.quaternion_norm_tolerance:
        raise ValueError(
            "root_rot is not unit length: maximum norm error is "
            f"{max_norm_error:.6g}, tolerance is {args.quaternion_norm_tolerance:.6g}"
        )
    root_rot = root_rot / quat_norm[:, None]

    # q and -q describe the same rotation, but keeping adjacent signs aligned
    # prevents interpolation from taking a discontinuous path.
    quaternion_sign_flips_fixed = 0
    for frame in range(1, frame_count):
        if float(np.dot(root_rot[frame - 1], root_rot[frame])) < 0.0:
            root_rot[frame] *= -1.0
            quaternion_sign_flips_fixed += 1

    csv_motion = np.concatenate((root_pos, root_rot, dof_pos), axis=1).astype(
        np.float32
    )
    if csv_motion.shape != (frame_count, 36):
        raise AssertionError(f"Unexpected CSV shape: {csv_motion.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, csv_motion, delimiter=",", fmt="%.9g")

    manifest = {
        "schema": "unitree_rl_mjlab_g1_29dof_csv_v1",
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "shape": [frame_count, 36],
            "dtype_before_text_encoding": "float32",
            "header": False,
        },
        "frames": frame_count,
        "input_fps_hz": fps,
        "duration_between_first_and_last_frame_s": (frame_count - 1) / fps,
        "root_quaternion_convention": "xyzw",
        "quaternion_max_norm_error_before_normalization": max_norm_error,
        "quaternion_sign_flips_fixed": quaternion_sign_flips_fixed,
        "joint_count": len(G1_29DOF_JOINT_NAMES),
        "joint_names_in_column_order": list(G1_29DOF_JOINT_NAMES),
        "unitree_rl_mjlab_commit": UNITREE_RL_MJLAB_COMMIT,
    }
    write_json(manifest_path, manifest)
    print(
        f"Exported {frame_count} frames at {fps:g} FPS to {output_path}\n"
        f"CSV shape: {csv_motion.shape}; manifest: {manifest_path}"
    )


def validate_csv(args: argparse.Namespace) -> None:
    path = args.input.resolve()
    motion = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if motion.ndim != 2 or motion.shape[1] != 36:
        raise ValueError(f"CSV must have 36 columns; got {motion.shape}")
    if motion.shape[0] < 2 or not np.isfinite(motion).all():
        raise ValueError("CSV must have at least two finite frames")
    norms = np.linalg.norm(motion[:, 3:7], axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > args.quaternion_norm_tolerance:
        raise ValueError(
            f"CSV quaternion maximum norm error {max_norm_error:.6g} exceeds "
            f"{args.quaternion_norm_tolerance:.6g}"
        )
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text())
        expected_hash = manifest["output"]["sha256"]
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"CSV SHA-256 mismatch: manifest has {expected_hash}, file has {actual_hash}"
            )
        if list(motion.shape) != manifest["output"]["shape"]:
            raise ValueError(
                f"CSV shape {list(motion.shape)} does not match manifest "
                f"{manifest['output']['shape']}"
            )
        if manifest["joint_names_in_column_order"] != list(G1_29DOF_JOINT_NAMES):
            raise ValueError("Manifest joint order does not match the expected G1 order")
    print(
        f"PASS: {path} has shape {motion.shape}, finite values, and XYZW unit "
        f"quaternions (max norm error {max_norm_error:.3g})."
    )


def expected_output_frames(source_frames: int, input_fps: float, output_fps: float) -> int:
    duration = (source_frames - 1) / input_fps
    return int(np.arange(0.0, duration, 1.0 / output_fps).shape[0])


def validate_npz(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else Path(str(input_path) + ".validation.json")
    )
    with np.load(input_path, allow_pickle=False) as archive:
        missing = [key for key in NPZ_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"NPZ is missing keys: {', '.join(missing)}")
        arrays = {key: np.asarray(archive[key]) for key in NPZ_KEYS}
        fps = scalar_fps(archive["fps"]) if "fps" in archive.files else args.fps

    frame_count = arrays["joint_pos"].shape[0]
    expected_shapes = {
        "joint_pos": (frame_count, 29),
        "joint_vel": (frame_count, 29),
        "body_pos_w": (frame_count, None, 3),
        "body_quat_w": (frame_count, None, 4),
        "body_lin_vel_w": (frame_count, None, 3),
        "body_ang_vel_w": (frame_count, None, 3),
    }
    body_count: int | None = None
    for key, array in arrays.items():
        expected = expected_shapes[key]
        if array.ndim != len(expected):
            raise ValueError(f"{key} expected {len(expected)} dimensions; got {array.shape}")
        for actual_dim, expected_dim in zip(array.shape, expected, strict=True):
            if expected_dim is not None and actual_dim != expected_dim:
                raise ValueError(f"{key} has invalid shape {array.shape}; expected {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or infinity")
        if key.startswith("body_"):
            if body_count is None:
                body_count = array.shape[1]
            elif body_count != array.shape[1]:
                raise ValueError("Body arrays do not share the same body count")

    body_quat_norm = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    max_quat_norm_error = float(np.max(np.abs(body_quat_norm - 1.0)))
    if max_quat_norm_error > args.quaternion_norm_tolerance:
        raise ValueError(
            f"body_quat_w maximum norm error {max_quat_norm_error:.6g} exceeds "
            f"{args.quaternion_norm_tolerance:.6g}"
        )

    source_check: dict[str, Any] | None = None
    if args.source_manifest is not None:
        source_manifest = json.loads(args.source_manifest.read_text())
        source_frames = int(source_manifest["frames"])
        input_fps = float(source_manifest["input_fps_hz"])
        expected_frames = expected_output_frames(source_frames, input_fps, fps)
        if frame_count != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} converted frames from source manifest, "
                f"got {frame_count}"
            )
        source_check = {
            "manifest": str(args.source_manifest.resolve()),
            "source_frames": source_frames,
            "source_fps_hz": input_fps,
            "expected_converted_frames": expected_frames,
        }

    report = {
        "status": "PASS",
        "schema": "unitree_rl_mjlab_tracking_npz_v1",
        "path": str(input_path),
        "sha256": sha256_file(input_path),
        "frames": frame_count,
        "fps_hz": fps,
        "joint_count": 29,
        "body_count": body_count,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "body_quaternion_max_norm_error": max_quat_norm_error,
        "all_values_finite": True,
        "source_check": source_check,
        "unitree_rl_mjlab_commit": UNITREE_RL_MJLAB_COMMIT,
    }
    write_json(report_path, report)
    print(
        f"PASS: {input_path} contains {frame_count} frames at {fps:g} FPS, "
        f"29 joints, and {body_count} bodies. Report: {report_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export a trusted GMR pickle to CSV")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--manifest", type=Path)
    export.add_argument("--expected-fps", type=float, default=30.0)
    export.add_argument("--quaternion-norm-tolerance", type=float, default=1.0e-4)
    export.set_defaults(func=export_pickle)

    csv_parser = subparsers.add_parser("validate-csv", help="Validate a MjLab CSV")
    csv_parser.add_argument("--input", type=Path, required=True)
    csv_parser.add_argument("--manifest", type=Path)
    csv_parser.add_argument("--quaternion-norm-tolerance", type=float, default=1.0e-4)
    csv_parser.set_defaults(func=validate_csv)

    npz_parser = subparsers.add_parser(
        "validate-npz", help="Validate an NPZ created by Unitree csv_to_npz.py"
    )
    npz_parser.add_argument("--input", type=Path, required=True)
    npz_parser.add_argument("--source-manifest", type=Path)
    npz_parser.add_argument("--report", type=Path)
    npz_parser.add_argument("--fps", type=float, default=50.0)
    npz_parser.add_argument("--quaternion-norm-tolerance", type=float, default=2.0e-4)
    npz_parser.set_defaults(func=validate_npz)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
