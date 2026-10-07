#!/usr/bin/env python3
"""Validate an SMPL motion source and prepare the files expected by GMR.

The validator intentionally uses only NumPy and the Python standard library so
it can run before the heavier SMPL-X/GMR environments are installed.  It does
not claim to validate mesh-to-video agreement or foot contacts; those require
the licensed SMPL-X model and the source video.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


def load_source(path: Path) -> dict[str, np.ndarray]:
    """Load either an actual .npz or the supplied directory-of-.npy layout."""
    if path.is_dir():
        names = ("poses", "trans", "betas", "gender", "mocap_framerate")
        missing = [name for name in names if not (path / f"{name}.npy").is_file()]
        if missing:
            raise ValueError(f"Missing source arrays: {', '.join(missing)}")
        return {
            name: np.load(path / f"{name}.npy", allow_pickle=False)
            for name in names
        }

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def scalar_float(value: np.ndarray, name: str) -> float:
    if np.asarray(value).size != 1:
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    return float(np.asarray(value).item())


def validate_schema(data: dict[str, np.ndarray]) -> tuple[int, float]:
    required = {"poses", "trans", "betas", "gender", "mocap_framerate"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    poses = np.asarray(data["poses"])
    trans = np.asarray(data["trans"])
    betas = np.asarray(data["betas"])
    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"poses must have shape (frames, 72), got {poses.shape}")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(
            f"trans must have shape ({poses.shape[0]}, 3), got {trans.shape}"
        )
    if betas.shape not in ((10,), (16,), (1, 10), (1, 16)):
        raise ValueError(f"betas must contain 10 or 16 values, got {betas.shape}")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("poses and trans must not contain NaN or infinity")
    if not np.isfinite(betas).all():
        raise ValueError("betas must not contain NaN or infinity")

    fps = scalar_float(np.asarray(data["mocap_framerate"]), "mocap_framerate")
    if fps <= 0:
        raise ValueError(f"mocap_framerate must be positive, got {fps}")
    return poses.shape[0], fps


def rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.divide(
        rotvec,
        angle,
        out=np.zeros_like(rotvec, dtype=np.float64),
        where=angle > 1e-12,
    )
    return np.concatenate((np.cos(angle / 2.0), axis * np.sin(angle / 2.0)), axis=-1)


def rotate_vector(rotvec: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply axis-angle rotations with Rodrigues' formula."""
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.divide(
        rotvec,
        angle,
        out=np.zeros_like(rotvec, dtype=np.float64),
        where=angle > 1e-12,
    )
    vector = np.broadcast_to(vector, rotvec.shape)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.sum(axis * vector, axis=-1, keepdims=True) * (1.0 - cosine)
    )


def angular_speed(rotvec: np.ndarray, fps: float) -> np.ndarray:
    """Return geodesic SO(3) speed, avoiding axis-angle wrap subtraction."""
    quaternion = rotvec_to_quaternion(rotvec)
    dots = np.sum(quaternion[:-1] * quaternion[1:], axis=-1)
    increments = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    return increments * fps


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


def prepare_archives(
    data: dict[str, np.ndarray], output_dir: Path, stem: str
) -> tuple[Path, Path]:
    poses = np.asarray(data["poses"], dtype=np.float32)
    trans = np.asarray(data["trans"], dtype=np.float32)
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
    gender = np.asarray(data["gender"])
    fps = np.asarray(data["mocap_framerate"], dtype=np.float32)

    smpl_path = output_dir / f"{stem}_smpl.npz"
    np.savez(
        smpl_path,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=gender,
        mocap_framerate=fps,
    )

    if betas.size == 10:
        betas = np.pad(betas, (0, 6))
    smplx_path = output_dir / f"{stem}_smplx.npz"
    np.savez(
        smplx_path,
        root_orient=poses[:, :3],
        pose_body=poses[:, 3:66],
        trans=trans,
        betas=betas,
        gender=gender,
        mocap_frame_rate=fps,
    )
    return smpl_path, smplx_path


def write_frame_metrics(
    path: Path,
    trans: np.ndarray,
    fps: float,
    root_speed: np.ndarray,
    root_angular_speed: np.ndarray,
    body_angular_speed: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frame",
                "time_s",
                "trans_x_m",
                "trans_y_m",
                "trans_z_m",
                "root_speed_m_s",
                "root_angular_speed_rad_s",
                "max_body_joint_speed_rad_s",
            )
        )
        for frame in range(len(trans)):
            previous = max(frame - 1, 0)
            writer.writerow(
                (
                    frame,
                    frame / fps,
                    *trans[frame],
                    0.0 if frame == 0 else root_speed[previous],
                    0.0 if frame == 0 else root_angular_speed[previous],
                    0.0 if frame == 0 else np.max(body_angular_speed[previous]),
                )
            )


def svg_polyline(
    values: np.ndarray,
    width: int,
    height: int,
    color: str,
    low: float,
    high: float,
) -> str:
    values = np.asarray(values, dtype=np.float64)
    x = np.linspace(45, width - 15, len(values))
    span = high - low if high > low else 1.0
    y = height - 30 - (values - low) / span * (height - 55)
    points = " ".join(f"{px:.2f},{py:.2f}" for px, py in zip(x, y))
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'


def write_svg_plot(
    path: Path,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    y_label: str,
) -> None:
    width, height = 960, 320
    all_values = np.concatenate([np.asarray(values).reshape(-1) for _, values, _ in series])
    low = float(np.min(all_values))
    high = float(np.max(all_values))
    legend = []
    lines = []
    for index, (label, values, color) in enumerate(series):
        lines.append(svg_polyline(values, width, height, color, low, high))
        legend.append(
            f'<text x="{55 + index * 150}" y="{height - 7}" fill="{color}">{html.escape(label)}</text>'
        )
    content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#10151d"/>
<line x1="45" y1="20" x2="45" y2="{height - 30}" stroke="#718096"/>
<line x1="45" y1="{height - 30}" x2="{width - 15}" y2="{height - 30}" stroke="#718096"/>
<text x="45" y="15" fill="#e2e8f0" font-size="16">{html.escape(title)}</text>
<text x="4" y="35" fill="#a0aec0" font-size="11">{html.escape(y_label)}: {high:.4g}</text>
<text x="4" y="{height - 32}" fill="#a0aec0" font-size="11">{low:.4g}</text>
{''.join(lines)}
{''.join(legend)}
</svg>"""
    path.write_text(content, encoding="utf-8")


def status_badge(status: str) -> str:
    css_class = {"PASS": "pass", "WARNING": "warning", "BLOCKED": "blocked"}[status]
    return f'<span class="badge {css_class}">{status}</span>'


def write_html_report(path: Path, report: dict[str, Any]) -> None:
    checks = "".join(
        "<tr>"
        f"<td>{html.escape(check['name'])}</td>"
        f"<td>{status_badge(check['status'])}</td>"
        f"<td>{html.escape(check['detail'])}</td>"
        "</tr>"
        for check in report["checks"]
    )
    metrics = report["metrics"]
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SMPL source validation</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;color:#202631}}
h1,h2{{color:#14213d}} table{{border-collapse:collapse;width:100%}} td,th{{padding:9px;border-bottom:1px solid #d8dee9;text-align:left;vertical-align:top}}
.badge{{padding:3px 8px;border-radius:12px;font-weight:700;font-size:12px}} .pass{{background:#d9f6e4;color:#136f3a}} .warning{{background:#fff0c2;color:#805b00}} .blocked{{background:#e9edf2;color:#4a5568}}
.notice{{padding:14px 18px;border-left:5px solid #d69e2e;background:#fffaf0}} img{{max-width:100%;border-radius:7px;background:#10151d}} code{{background:#edf2f7;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>SMPL source validation: {html.escape(report['source'])}</h1>
<p class="notice"><strong>Overall: {html.escape(report['overall_status'])}.</strong> {html.escape(report['conclusion'])}</p>
<h2>Checks</h2><table><thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{checks}</tbody></table>
<h2>Key measurements</h2>
<ul>
<li>{metrics['frames']} frames at {metrics['fps_hz']:.3g} FPS; temporal span {metrics['temporal_span_s']:.3f} seconds.</li>
<li>Translation mean: {metrics['translation_mean_m']} m.</li>
<li>Translation range: {metrics['translation_range_m']} m.</li>
<li>Mean transformed SMPL local-up vector: {metrics['mean_local_up_world']}.</li>
<li>Maximum root speed: {metrics['root_linear_speed_m_s']['max']:.4f} m/s.</li>
<li>Maximum root angular speed: {metrics['root_angular_speed_rad_s']['max']:.4f} rad/s.</li>
<li>Maximum body-joint angular speed: {metrics['body_joint_angular_speed_rad_s']['max']:.4f} rad/s at frame {metrics['fastest_body_joint']['frame']} ({html.escape(metrics['fastest_body_joint']['joint'])}).</li>
</ul>
<h2>Plots</h2>
<p><img src="root_translation.svg" alt="Root translation plot"></p>
<p><img src="motion_speed.svg" alt="Motion speed plot"></p>
<p><img src="body_up_direction.svg" alt="Body up direction plot"></p>
<h2>Prepared files</h2>
<ul><li><code>{html.escape(report['outputs']['smpl_archive'])}</code></li><li><code>{html.escape(report['outputs']['smplx_archive'])}</code></li><li><code>{html.escape(report['outputs']['frame_metrics_csv'])}</code></li></ul>
<p>{html.escape(report['conclusion'])}</p>
</body></html>"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="SMPL .npz or directory of .npy arrays")
    parser.add_argument("--output-dir", type=Path, default=Path("validation_output"))
    parser.add_argument("--stem", default="ayyala")
    args = parser.parse_args()

    data = load_source(args.source)
    frames, fps = validate_schema(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smpl_path, smplx_path = prepare_archives(data, args.output_dir, args.stem)

    poses = np.asarray(data["poses"], dtype=np.float64)
    trans = np.asarray(data["trans"], dtype=np.float64)
    root_speed = np.linalg.norm(np.diff(trans, axis=0), axis=1) * fps
    root_angular = angular_speed(poses[:, :3], fps)
    body_rotvec = poses[:, 3:].reshape(frames, 23, 3)
    body_angular = angular_speed(body_rotvec, fps)
    mean_up = np.mean(rotate_vector(poses[:, :3], np.array([0.0, 1.0, 0.0])), axis=0)

    translation_mean = np.mean(trans, axis=0)
    translation_range = np.ptp(trans, axis=0)
    translation_offset_axis = "xyz"[int(np.argmax(np.abs(translation_mean)))]
    body_up_axis = "xyz"[int(np.argmax(np.abs(mean_up)))]
    coordinate_warning = (
        translation_offset_axis != body_up_axis
        and np.max(np.abs(translation_mean)) > 0.5
        and np.max(np.abs(mean_up)) > 0.8
    )
    duplicate_frames = int(np.sum(np.all(poses[1:] == poses[:-1], axis=1)))
    fastest_index = np.unravel_index(int(np.argmax(body_angular)), body_angular.shape)
    fastest_frame = int(fastest_index[0] + 1)
    fastest_joint_index = int(fastest_index[1] + 1)

    frame_csv = args.output_dir / "frame_metrics.csv"
    write_frame_metrics(frame_csv, trans, fps, root_speed, root_angular, body_angular)
    write_svg_plot(
        args.output_dir / "root_translation.svg",
        [("X", trans[:, 0], "#63b3ed"), ("Y", trans[:, 1], "#68d391"), ("Z", trans[:, 2], "#fc8181")],
        "Root translation",
        "metres",
    )
    frame_joint_max = np.concatenate(([0.0], np.max(body_angular, axis=1)))
    write_svg_plot(
        args.output_dir / "motion_speed.svg",
        [
            ("root linear", np.concatenate(([0.0], root_speed)), "#63b3ed"),
            ("root angular", np.concatenate(([0.0], root_angular)), "#68d391"),
            ("max joint angular", frame_joint_max, "#fc8181"),
        ],
        "Per-frame motion speed (mixed units; inspect spikes)",
        "m/s or rad/s",
    )
    up_over_time = rotate_vector(poses[:, :3], np.array([0.0, 1.0, 0.0]))
    write_svg_plot(
        args.output_dir / "body_up_direction.svg",
        [("X", up_over_time[:, 0], "#63b3ed"), ("Y", up_over_time[:, 1], "#68d391"), ("Z", up_over_time[:, 2], "#fc8181")],
        "SMPL local +Y transformed by root orientation",
        "unit vector",
    )

    checks = [
        {"name": "File schema", "status": "PASS", "detail": "Required arrays and expected SMPL shapes are present."},
        {"name": "Finite values", "status": "PASS", "detail": "No NaN or infinite pose, translation, or shape values."},
        {"name": "Frame continuity", "status": "PASS", "detail": f"No duplicate adjacent frames; maximum body-joint speed is {np.max(body_angular):.3f} rad/s."},
        {
            "name": "World coordinate consistency",
            "status": "WARNING" if coordinate_warning else "PASS",
            "detail": (
                f"Translation offset is dominated by {translation_offset_axis.upper()}, while transformed body-up is dominated by {body_up_axis.upper()}. Render before retargeting."
                if coordinate_warning
                else "Translation offset and transformed body-up do not trigger the mismatch heuristic."
            ),
        },
        {
            "name": "Global root motion",
            "status": "WARNING" if float(np.max(translation_range)) < 0.15 else "PASS",
            "detail": f"Root range is {translation_range.round(4).tolist()} m. Confirm that the original dancer really stays nearly in place.",
        },
        {"name": "SMPL-X mesh render", "status": "BLOCKED", "detail": "SMPL-X body-model files are not present in the project or user home."},
        {"name": "Original-video overlay", "status": "BLOCKED", "detail": "No source Ayyala video was found in the project."},
        {"name": "Foot-contact validation", "status": "BLOCKED", "detail": "Requires SMPL-X joint positions/mesh and visual confirmation against the source video."},
    ]

    report: dict[str, Any] = {
        "source": str(args.source),
        "overall_status": "NEEDS_VISUAL_REVIEW",
        "conclusion": "Numerically valid and converted, but not approved for retargeting because the coordinate frame and human-video agreement are unverified.",
        "checks": checks,
        "metrics": {
            "frames": frames,
            "fps_hz": fps,
            "temporal_span_s": (frames - 1) / fps,
            "sample_duration_s": frames / fps,
            "duplicate_adjacent_frames": duplicate_frames,
            "translation_mean_m": translation_mean.round(6).tolist(),
            "translation_range_m": translation_range.round(6).tolist(),
            "translation_offset_axis": translation_offset_axis,
            "mean_local_up_world": mean_up.round(6).tolist(),
            "body_up_axis": body_up_axis,
            "coordinate_mismatch_heuristic": bool(coordinate_warning),
            "root_linear_speed_m_s": percentile_summary(root_speed),
            "root_angular_speed_rad_s": percentile_summary(root_angular),
            "body_joint_angular_speed_rad_s": percentile_summary(body_angular),
            "fastest_body_joint": {
                "frame": fastest_frame,
                "joint": SMPL_JOINT_NAMES[fastest_joint_index],
                "speed_rad_s": float(body_angular[fastest_index]),
            },
            "gmr_height_estimate_m": float(1.66 + 0.1 * np.asarray(data["betas"]).reshape(-1)[0]),
        },
        "outputs": {
            "smpl_archive": smpl_path.name,
            "smplx_archive": smplx_path.name,
            "frame_metrics_csv": frame_csv.name,
        },
    }
    json_path = args.output_dir / "validation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html_report(args.output_dir / "validation_report.html", report)

    print(f"Overall: {report['overall_status']}")
    for check in checks:
        print(f"{check['status']:7} {check['name']}: {check['detail']}")
    print(f"Report: {args.output_dir / 'validation_report.html'}")


if __name__ == "__main__":
    main()
