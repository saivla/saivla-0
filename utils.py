"""Evaluation utilities: video saving, summary output, quaternion conversion, etc."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import imageio
import numpy as np


# ------------------------------------------------------------------
# Quaternion -> Axis-angle (pure math, no model dependency)
# ------------------------------------------------------------------

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to axis-angle representation (ax, ay, az).

    Numerically consistent with ``quat2axisangle_numpy`` in eval_Sai0_1.py.
    """
    qw = float(quat[3])
    qw = max(-1.0, min(1.0, qw))
    den = math.sqrt(1.0 - qw * qw)
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(qw)
    return (quat[:3].astype(np.float32) * angle) / den


# ------------------------------------------------------------------
# Extract state vector from LIBERO obs
# ------------------------------------------------------------------

def extract_state_from_obs(obs: dict) -> np.ndarray:
    """
    Extract an 8-dim state vector from a LIBERO observation dict::

        [gripper_0, gripper_1, x, y, z, ax, ay, az]

    The server handles further index extraction and normalization;
    the client only needs to send this raw value.
    """
    gripper = obs["robot0_gripper_qpos"]
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    return np.array(
        [gripper[0], gripper[1], xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]],
        dtype=np.float32,
    )


# ------------------------------------------------------------------
# Extract images from LIBERO obs
# ------------------------------------------------------------------

def extract_images_from_obs(
    obs: dict,
    resize_to: Optional[int] = None,
    flip: bool = False,
) -> tuple:
    """
    Return (agentview_img, wrist_img), both as uint8 RGB ndarrays.

    Args:
        flip: Whether to flip 180 degrees (to match model view).
    """
    img = obs["agentview_image"].copy()
    wrist = obs["robot0_eye_in_hand_image"].copy()

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if wrist.dtype != np.uint8:
        wrist = (wrist * 255).astype(np.uint8) if wrist.max() <= 1.0 else wrist.astype(np.uint8)

    if resize_to is not None:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        wrist = cv2.resize(wrist, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)

    if flip:
        img = np.ascontiguousarray(img[::-1, ::-1, :])
        wrist = np.ascontiguousarray(wrist[::-1, ::-1, :])

    return img, wrist


# ------------------------------------------------------------------
# Video saving
# ------------------------------------------------------------------

def save_rollout_video(
    top_frames: List[np.ndarray],
    wrist_frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
) -> Optional[str]:
    """Compose agentview + wrist side-by-side into an mp4. Returns path on success, None on failure."""
    if not top_frames or not wrist_frames:
        return None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
        )
        for t, w in zip(top_frames, wrist_frames):
            t = t.astype(np.uint8) if t.dtype != np.uint8 else t
            w = w.astype(np.uint8) if w.dtype != np.uint8 else w
            writer.append_data(np.hstack((t, w)))
        writer.close()
        return output_path
    except Exception as e:
        print(f"Video save failed: {e}")
        return None


# ------------------------------------------------------------------
# Evaluation summary
# ------------------------------------------------------------------

def save_summary(
    task_suite: str,
    task_results: Dict[int, dict],
    total_episodes: int,
    total_successes: int,
    server_version: str,
    output_dir: str,
) -> str:
    """Write evaluation results as JSON and return the file path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "task_suite": task_suite,
        "overall_success_rate": round(total_successes / max(total_episodes, 1), 4),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "per_task": [
            {
                "task_id": tid,
                "description": info["description"],
                "success_rate": info["success_rate"],
                "successes": info["successes"],
                "episodes": info["episodes"],
            }
            for tid, info in sorted(task_results.items())
        ],
        "server_version": server_version,
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = out / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return str(path)
