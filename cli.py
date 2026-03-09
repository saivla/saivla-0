"""
sai0-eval CLI entry point

After installation, run directly::

    sai0-eval --server https://api.sai0.ai --api-key sk-xxx --task-suite libero_spatial
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="sai0-eval",
        description="Sai0-VLA LIBERO remote evaluation client",
    )

    # Connection
    parser.add_argument(
        "--server",
        type=str,
        default=os.environ.get("SAI0_SERVER", "http://localhost:5000"),
        help="Sai0-VLA inference server URL (or set SAI0_SERVER env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("SAI0_API_KEY"),
        help="API Key (or set SAI0_API_KEY env var)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--task-suite",
        type=str,
        default="libero_spatial",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="LIBERO task suite",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per task")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per trial")
    parser.add_argument(
        "--action-chunk-exec",
        type=int,
        default=16,
        help="Actions to execute per API call (default 16, matches eval script)",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Task IDs to evaluate (comma-separated, e.g. 0,1,2)",
    )
    parser.add_argument("--max-tasks", type=int, default=-1, help="Max number of tasks to evaluate")
    parser.add_argument("--resolution", type=int, default=256, help="Environment image resolution")
    parser.add_argument("--env-seed", type=int, default=None, help="Environment random seed")

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./eval_results", help="Output directory for results"
    )
    parser.add_argument(
        "--save-video", action="store_true", default=True, help="Save evaluation videos"
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Do not save videos"
    )
    parser.add_argument(
        "--no-flip-video", action="store_true",
        help="Do not flip video frames 180 degrees (default: flip to match model view)"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    # Quick commands
    parser.add_argument(
        "--check", action="store_true", help="Only check server connection, skip evaluation"
    )

    args = parser.parse_args()

    # --check mode
    if args.check:
        from .client import Sai0VLAClient

        client = Sai0VLAClient(args.server, api_key=args.api_key)
        try:
            ver = client.version()
            health = client.health()
            print(f"Server: {args.server}")
            print(f"Version: {ver}")
            print(f"Health: {health}")
            sys.exit(0)
        except Exception as e:
            print(f"Connection failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Parse task_ids
    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]

    from .libero_eval import run_libero_eval

    run_libero_eval(
        server_url=args.server,
        api_key=args.api_key,
        task_suite=args.task_suite,
        num_trials=args.trials,
        max_steps=args.max_steps,
        action_chunk_exec=args.action_chunk_exec,
        resolution=args.resolution,
        env_seed=args.env_seed,
        task_ids=task_ids,
        max_tasks=args.max_tasks,
        output_dir=args.output_dir,
        save_video=not args.no_video,
        flip_video=not args.no_flip_video,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
