"""
LIBERO Remote Evaluation Script

Run LIBERO benchmark evaluation via the Sai0-VLA API.
Users only need a local LIBERO environment -- no model code required.

Usage::

    from .libero_eval import run_libero_eval
    run_libero_eval(
        server_url="https://api.sai0.ai",
        api_key="sk-xxx",
        task_suite="libero_spatial",
    )
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tqdm

from .client import Sai0VLAClient
from .utils import (
    extract_images_from_obs,
    extract_state_from_obs,
    save_rollout_video,
    save_summary,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# LIBERO lazy imports
# ------------------------------------------------------------------

_benchmark = None
_get_libero_path = None
_OffScreenRenderEnv = None


def _fix_robosuite_log():
    """Redirect robosuite log if /tmp/robosuite.log is not writable."""
    import logging as _logging

    try:
        with open("/tmp/robosuite.log", "a"):
            return
    except PermissionError:
        pass

    user_log = os.path.expanduser("~/.robosuite/robosuite.log")
    os.makedirs(os.path.dirname(user_log), exist_ok=True)
    _orig = _logging.FileHandler

    class _Patched(_orig):
        def __init__(self, filename, mode="a", encoding=None, delay=False):
            if filename == "/tmp/robosuite.log":
                filename = user_log
            super().__init__(filename, mode, encoding, delay)

    _logging.FileHandler = _Patched


def _init_libero():
    global _benchmark, _get_libero_path, _OffScreenRenderEnv
    if _benchmark is not None:
        return

    _fix_robosuite_log()

    try:
        from libero.libero import benchmark as bm, get_libero_path as glp
        from libero.libero.envs import OffScreenRenderEnv as Env
    except ImportError as e:
        raise RuntimeError(
            "LIBERO is not installed. Please run:\n"
            "  pip install robosuite==1.4.0 libero\n"
            f"Original error: {e}"
        )

    _benchmark = bm
    _get_libero_path = glp
    _OffScreenRenderEnv = Env


def _make_env(task, resolution: int = 256, seed: Optional[int] = None):
    bddl = os.path.join(
        _get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = _OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed if seed is not None else random.randint(0, 2**31 - 1))
    return env


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def run_libero_eval(
    server_url: str,
    api_key: Optional[str] = None,
    task_suite: str = "libero_spatial",
    num_trials: int = 10,
    max_steps: int = 600,
    num_steps_wait: int = 10,
    resolution: int = 256,
    action_chunk_exec: int = 16,
    env_seed: Optional[int] = None,
    task_ids: Optional[List[int]] = None,
    max_tasks: int = -1,
    output_dir: str = "./eval_results",
    save_video: bool = True,
    flip_video: bool = True,
    verbose: bool = True,
):
    """
    Run LIBERO remote evaluation.

    Args:
        server_url: Sai0-VLA inference server address.
        api_key: API Key (can be None if server auth is disabled).
        task_suite: LIBERO task suite name.
        num_trials: Number of trials per task.
        max_steps: Max steps per trial.
        num_steps_wait: Steps to wait for environment to stabilize.
        resolution: Environment image resolution.
        action_chunk_exec: Actions to execute per API call (default 16 = all predicted).
        env_seed: Environment random seed (None = random).
        task_ids: List of task IDs to evaluate (None = all).
        max_tasks: Max number of tasks to evaluate (-1 = no limit).
        output_dir: Output directory for results.
        save_video: Whether to save evaluation videos.
        flip_video: Whether to flip video frames 180 degrees (match model view, default True).
        verbose: Whether to print detailed progress.
    """
    _init_libero()

    out = Path(output_dir)
    video_dir = out / "videos"
    out.mkdir(parents=True, exist_ok=True)
    if save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client and check connection
    client = Sai0VLAClient(server_url, api_key=api_key)
    try:
        ver = client.version()
        server_ver = ver.get("version", "unknown")
    except Exception as e:
        raise ConnectionError(f"Cannot connect to server {server_url}: {e}")

    if verbose:
        print(f"Connected to Sai0-VLA server v{server_ver} ({server_url})")

    # Initialize LIBERO tasks
    bench = _benchmark.get_benchmark_dict()
    suite = bench[task_suite]()
    n_tasks = suite.n_tasks

    if task_ids is not None:
        id_list = task_ids
    elif max_tasks > 0:
        id_list = list(range(min(max_tasks, n_tasks)))
    else:
        id_list = list(range(n_tasks))

    print(f"\n{'=' * 60}")
    print(f"LIBERO Remote Evaluation (sai0-vla-client)")
    print(f"{'=' * 60}")
    print(f"  Server:       {server_url} (v{server_ver})")
    print(f"  Task Suite:   {task_suite} ({len(id_list)}/{n_tasks} tasks)")
    print(f"  Trials/task:  {num_trials}")
    print(f"  Max steps:    {max_steps}")
    print(f"  Chunk exec:   {action_chunk_exec}")
    print(f"  Output dir:   {output_dir}")
    print(f"{'=' * 60}\n")

    # Statistics
    task_results: Dict[int, dict] = {}
    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(id_list, desc="Tasks"):
        task = suite.get_task(task_id)
        task_description = task.language
        initial_states = suite.get_task_init_states(task_id)

        if verbose:
            print(f"\n[Task {task_id}] {task_description}")

        env = _make_env(task, resolution=resolution, seed=env_seed)
        task_successes = 0

        for trial in range(num_trials):
            trial_t0 = time.time()
            if verbose:
                print(f"  Trial {trial + 1}/{num_trials}", end=" ")

            env.reset()
            state_idx = trial % initial_states.shape[0]
            obs = env.set_init_state(initial_states[state_idx])

            # Wait for objects to settle
            for _ in range(num_steps_wait):
                obs, _, _, _ = env.step(DUMMY_ACTION)

            # Rollout
            done = False
            success = False
            top_frames, wrist_frames = [], []
            action_queue: List[np.ndarray] = []
            step = 0

            for step in range(max_steps):
                if done:
                    break

                # Record frames (flip to match model view)
                if save_video:
                    img_v, wrist_v = extract_images_from_obs(obs, flip=flip_video)
                    top_frames.append(img_v)
                    wrist_frames.append(wrist_v)

                # If action queue is empty, call API
                if len(action_queue) == 0:
                    img, wrist_img = extract_images_from_obs(obs)
                    state = extract_state_from_obs(obs)

                    t_api = time.time()
                    actions = client.act(
                        images=[img, wrist_img],
                        state=state,
                        instruction=task_description,
                        task_suite=task_suite,
                    )
                    api_ms = (time.time() - t_api) * 1000

                    n_exec = min(action_chunk_exec, len(actions))
                    for i in range(n_exec):
                        action = actions[i].copy()
                        action[-1] = np.sign(action[-1])
                        action_queue.append(action)

                # Execute one action step
                action = action_queue.pop(0)
                obs, reward, done, info = env.step(action.tolist())

                if done and reward > 0:
                    success = True

            if success:
                task_successes += 1
                total_successes += 1
            total_episodes += 1

            trial_time = time.time() - trial_t0
            status = "OK" if success else "FAIL"
            if verbose:
                rate = total_successes / total_episodes * 100
                print(
                    f"  {status}  steps={step+1}  time={trial_time:.1f}s  "
                    f"running={rate:.1f}%"
                )

            # Save video
            if save_video and top_frames:
                tag = "success" if success else "fail"
                safe_desc = task_description[:40].replace(" ", "_").lower()
                vpath = (
                    video_dir
                    / f"task{task_id}"
                    / f"trial{trial}_{tag}_{safe_desc}.mp4"
                )
                save_rollout_video(top_frames, wrist_frames, str(vpath))

        env.close()

        sr = task_successes / max(num_trials, 1)
        task_results[task_id] = {
            "description": task_description,
            "success_rate": round(sr, 4),
            "successes": task_successes,
            "episodes": num_trials,
        }
        if verbose:
            print(f"  => Task {task_id} success rate: {sr:.1%}")

    # Print summary
    overall = total_successes / max(total_episodes, 1)
    print(f"\n{'=' * 60}")
    print(f"Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"  Task Suite:         {task_suite}")
    print(f"  Overall Success:    {total_successes}/{total_episodes} ({overall:.1%})")
    for tid, r in sorted(task_results.items()):
        print(f"  Task {tid:>2d}: {r['success_rate']:.1%}  {r['description'][:50]}")
    print(f"{'=' * 60}")

    summary_path = save_summary(
        task_suite=task_suite,
        task_results=task_results,
        total_episodes=total_episodes,
        total_successes=total_successes,
        server_version=server_ver,
        output_dir=output_dir,
    )
    print(f"\nResults saved to: {summary_path}")
    return task_results
