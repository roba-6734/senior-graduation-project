#!/usr/bin/env python3
"""Compare an SMPL-X motion with the source video and render an overlay."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import smplx
import torch


JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

MEDIAPIPE = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot": 31,
    "right_foot": 32,
}

MATCH_NAMES = tuple(MEDIAPIPE)
EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
)


def reconstruct_joints(motion_path: Path, model_root: Path, chunk_size: int = 64) -> np.ndarray:
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    frames = len(motion["root_orient"])
    betas = np.asarray(motion["betas"], dtype=np.float32).reshape(-1)
    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender=str(np.asarray(motion["gender"]).item()),
        ext="pkl",
        use_pca=False,
        num_betas=len(betas),
        batch_size=1,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, frames, chunk_size):
            stop = min(start + chunk_size, frames)
            count = stop - start
            zeros3 = torch.zeros(count, 3)
            output = model(
                betas=torch.from_numpy(np.repeat(betas[None], count, axis=0)),
                global_orient=torch.from_numpy(motion["root_orient"][start:stop]),
                body_pose=torch.from_numpy(motion["pose_body"][start:stop]),
                transl=torch.from_numpy(motion["trans"][start:stop]),
                left_hand_pose=torch.zeros(count, 45),
                right_hand_pose=torch.zeros(count, 45),
                jaw_pose=zeros3,
                leye_pose=zeros3,
                reye_pose=zeros3,
                expression=torch.zeros(count, 10),
                return_verts=False,
            )
            chunks.append(output.joints[:, :22].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def detect_video_landmarks(
    video_path: Path, task_path: Path
) -> tuple[np.ndarray, dict[str, float | int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detected: list[np.ndarray] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            result = landmarker.detect_for_video(image, timestamp_ms)
            values = np.full((33, 4), np.nan, dtype=np.float32)
            if result.pose_landmarks:
                for index, landmark in enumerate(result.pose_landmarks[0]):
                    values[index] = (
                        landmark.x * width,
                        landmark.y * height,
                        landmark.visibility,
                        landmark.presence,
                    )
            detected.append(values)
            frame_index += 1
    capture.release()
    return np.asarray(detected), {
        "fps": fps,
        "width": width,
        "height": height,
        "expected_frames": expected_frames,
        "decoded_frames": len(detected),
    }


def projection_coordinates(joints: np.ndarray, perspective: bool) -> np.ndarray:
    x = joints[..., 0]
    z = joints[..., 2]
    if perspective:
        depth = np.clip(joints[..., 1], 0.2, None)
        x = x / depth
        z = z / depth
    return np.stack((x, -z), axis=-1)


def fit_similarity(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    valid = np.isfinite(source).all(axis=-1) & np.isfinite(target).all(axis=-1) & (weights > 0)
    q = source[valid]
    p = target[valid]
    w = weights[valid, None]
    if len(q) < 4:
        return float("nan"), np.full(2, np.nan)
    q_mean = np.sum(w * q, axis=0) / np.sum(w)
    p_mean = np.sum(w * p, axis=0) / np.sum(w)
    q_centered = q - q_mean
    p_centered = p - p_mean
    denominator = np.sum(w * q_centered * q_centered)
    scale = float(np.sum(w * q_centered * p_centered) / denominator)
    translation = p_mean - scale * q_mean
    return scale, translation


def fit_affine_camera(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one fixed affine 3D-to-2D camera over the complete sequence."""
    valid = np.isfinite(source).all(axis=-1) & np.isfinite(target).all(axis=-1) & (weights > 0)
    design = np.concatenate((source, np.ones((*source.shape[:-1], 1))), axis=-1)
    design = design[valid]
    observations = target[valid]
    root_weights = np.sqrt(weights[valid, None])
    matrix = np.linalg.lstsq(
        design * root_weights, observations * root_weights, rcond=None
    )[0]
    return matrix, np.concatenate((source, np.ones((*source.shape[:-1], 1))), axis=-1) @ matrix


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def timing_lag(model_points: np.ndarray, observed_points: np.ndarray, valid: np.ndarray) -> tuple[int, float]:
    model_center = (model_points[:, MATCH_NAMES.index("left_hip")] + model_points[:, MATCH_NAMES.index("right_hip")]) / 2
    observed_center = (observed_points[:, MATCH_NAMES.index("left_hip")] + observed_points[:, MATCH_NAMES.index("right_hip")]) / 2
    model_relative = model_points - model_center[:, None]
    observed_relative = observed_points - observed_center[:, None]
    model_velocity = np.linalg.norm(np.diff(model_relative, axis=0), axis=-1)
    observed_velocity = np.linalg.norm(np.diff(observed_relative, axis=0), axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    model_signal = np.nanmean(np.where(pair_valid, model_velocity, np.nan), axis=1)
    observed_signal = np.nanmean(np.where(pair_valid, observed_velocity, np.nan), axis=1)
    model_signal = smooth(np.nan_to_num(model_signal, nan=np.nanmedian(model_signal)))
    observed_signal = smooth(np.nan_to_num(observed_signal, nan=np.nanmedian(observed_signal)))
    best_lag, best_correlation = 0, -1.0
    for lag in range(-15, 16):
        if lag < 0:
            first, second = model_signal[-lag:], observed_signal[:lag]
        elif lag > 0:
            first, second = model_signal[:-lag], observed_signal[lag:]
        else:
            first, second = model_signal, observed_signal
        correlation = float(np.corrcoef(first, second)[0, 1])
        if correlation > best_correlation:
            best_lag, best_correlation = lag, correlation
    return best_lag, best_correlation


def draw_skeleton(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    point_by_name = {name: points[index] for index, name in enumerate(MATCH_NAMES)}
    for first, second in EDGES:
        p1, p2 = point_by_name[first], point_by_name[second]
        if np.isfinite(p1).all() and np.isfinite(p2).all():
            cv2.line(frame, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), color, thickness, cv2.LINE_AA)
    for point in points:
        if np.isfinite(point).all():
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)


def render_overlay(
    video_path: Path,
    output_path: Path,
    model_points: np.ndarray,
    observed_points: np.ndarray,
    valid: np.ndarray,
    errors: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    temporary = output_path.with_name(output_path.stem + "_silent.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        observed = observed_points[frame_index].copy()
        observed[~valid[frame_index]] = np.nan
        draw_skeleton(frame, observed, (255, 0, 255), 3)
        draw_skeleton(frame, model_points[frame_index], (255, 255, 0), 2)
        mean_error = float(np.nanmean(errors[frame_index]))
        cv2.rectangle(frame, (8, 8), (330, 76), (0, 0, 0), -1)
        cv2.putText(frame, "SMPL-X", (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Video landmarks", (105, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_index:03d}  error {mean_error:.1f}px", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(temporary), "-i", str(video_path),
                "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
                "-crf", "20", "-preset", "medium", "-c:a", "copy",
                "-shortest", str(output_path),
            ],
            check=True,
        )
        temporary.unlink()
    else:
        temporary.rename(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--pose-task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("validation_output/video_alignment"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joints = reconstruct_joints(args.motion, args.model_root)
    landmarks, video = detect_video_landmarks(args.video, args.pose_task)
    if len(joints) != len(landmarks):
        raise ValueError(f"Motion/video frame mismatch: {len(joints)} != {len(landmarks)}")

    smpl_indices = np.asarray([JOINTS[name] for name in MATCH_NAMES])
    mp_indices = np.asarray([MEDIAPIPE[name] for name in MATCH_NAMES])
    matched_joints = joints[:, smpl_indices]
    observed = landmarks[:, mp_indices, :2]
    confidence = landmarks[:, mp_indices, 2] * landmarks[:, mp_indices, 3]
    valid = np.isfinite(observed).all(axis=-1) & (confidence >= 0.5)
    weights = np.where(valid, confidence, 0.0)

    candidates: dict[str, dict[str, object]] = {}
    for name, perspective in (("perspective_x_over_y", True), ("orthographic_x_z", False)):
        projected = projection_coordinates(matched_joints, perspective)
        scale, translation = fit_similarity(projected, observed, weights)
        pixels = scale * projected + translation
        error = np.linalg.norm(pixels - observed, axis=-1)
        masked_error = np.where(valid, error, np.nan)
        candidates[name] = {
            "perspective": perspective,
            "scale": scale,
            "translation": translation,
            "points": pixels,
            "error": masked_error,
            "mean_error": float(np.nanmean(masked_error)),
        }
    affine_matrix, affine_pixels = fit_affine_camera(matched_joints, observed, weights)
    affine_error = np.linalg.norm(affine_pixels - observed, axis=-1)
    affine_masked_error = np.where(valid, affine_error, np.nan)
    candidates["global_affine_camera"] = {
        "points": affine_pixels,
        "error": affine_masked_error,
        "mean_error": float(np.nanmean(affine_masked_error)),
        "affine_matrix": affine_matrix,
    }
    selected_name = min(candidates, key=lambda key: candidates[key]["mean_error"])
    selected = candidates[selected_name]
    model_pixels = np.asarray(selected["points"])
    errors = np.asarray(selected["error"])

    per_frame_error = []
    for frame in range(len(joints)):
        frame_weights = weights[frame]
        scale, translation = fit_similarity(
            model_pixels[frame],
            observed[frame],
            frame_weights,
        )
        aligned = scale * model_pixels[frame] + translation
        current = np.linalg.norm(aligned - observed[frame], axis=-1)
        per_frame_error.extend(current[valid[frame]].tolist())

    angle_errors: list[float] = []
    name_to_index = {name: index for index, name in enumerate(MATCH_NAMES)}
    for first, second in EDGES:
        first_index, second_index = name_to_index[first], name_to_index[second]
        edge_valid = valid[:, first_index] & valid[:, second_index]
        a = model_pixels[:, second_index] - model_pixels[:, first_index]
        b = observed[:, second_index] - observed[:, first_index]
        cosine = np.sum(a * b, axis=-1) / np.clip(
            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8, None
        )
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        angle_errors.extend(angle[edge_valid].tolist())

    best_lag, lag_correlation = timing_lag(model_pixels, observed, valid)
    diagonal = float(np.hypot(video["width"], video["height"]))
    detection_ratio = float(np.mean(np.isfinite(landmarks[:, 0, 0])))
    z_extent = joints[:, :, 2].max(axis=1) - joints[:, :, 2].min(axis=1)
    x_extent = joints[:, :, 0].max(axis=1) - joints[:, :, 0].min(axis=1)
    y_extent = joints[:, :, 1].max(axis=1) - joints[:, :, 1].min(axis=1)

    mean_error = float(np.nanmean(errors))
    mean_angle_error = float(np.mean(angle_errors))
    normalized_mean_error = mean_error / diagonal * 100
    joint_error = {
        name: float(np.nanmean(errors[:, index]))
        for index, name in enumerate(MATCH_NAMES)
    }
    report = {
        "status": (
            "PASS"
            if detection_ratio > 0.95
            and normalized_mean_error < 3.0
            and mean_angle_error < 15.0
            and abs(best_lag) <= 2
            else "REVIEW"
        ),
        "video": {**video, "path": str(args.video)},
        "motion_frames": len(joints),
        "pose_detection_frame_ratio": detection_ratio,
        "selected_projection": selected_name,
        "affine_camera_matrix": affine_matrix.round(6).tolist(),
        "candidate_mean_error_px": {
            name: float(values["mean_error"]) for name, values in candidates.items()
        },
        "global_reprojection_error_px": {
            "mean": mean_error,
            "p50": float(np.nanpercentile(errors, 50)),
            "p95": float(np.nanpercentile(errors, 95)),
            "normalized_mean_percent_diagonal": normalized_mean_error,
            "mean_by_joint": joint_error,
        },
        "per_frame_similarity_pose_error_px": {
            "mean": float(np.mean(per_frame_error)),
            "p95": float(np.percentile(per_frame_error, 95)),
        },
        "limb_direction_error_degrees": {
            "mean": mean_angle_error,
            "p95": float(np.percentile(angle_errors, 95)),
        },
        "temporal_alignment": {
            "best_lag_frames": best_lag,
            "best_lag_seconds": best_lag / float(video["fps"]),
            "motion_speed_correlation": lag_correlation,
        },
        "coordinate_evidence": {
            "mean_body_extent_x_m": float(np.mean(x_extent)),
            "mean_body_extent_y_m": float(np.mean(y_extent)),
            "mean_body_extent_z_m": float(np.mean(z_extent)),
            "interpretation": "SMPL-X body height is aligned with Z; Y behaves as camera depth. The earlier Y/Z warning was caused by treating translation magnitude as height.",
        },
    }
    np.save(args.output_dir / "smplx_joints.npy", joints)
    np.savez_compressed(args.output_dir / "video_landmarks.npz", landmarks=landmarks)
    (args.output_dir / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_overlay(
        args.video,
        args.output_dir / "source_alignment_overlay.mp4",
        model_pixels,
        observed,
        valid,
        errors,
        float(video["fps"]),
        int(video["width"]),
        int(video["height"]),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
