import argparse
import sys

from dotenv import load_dotenv

load_dotenv(override=True)


def cli():
    parser = argparse.ArgumentParser(
        description="BATNA-aware RL for negotiation",
        usage="python main.py {train,eval} [options]",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Run GRPO training")
    train_parser.add_argument(
        "--config", type=str, default="configs/train.yaml", help="YAML config file"
    )

    train_parser.add_argument("--seed", type=int, default=None, help="Override the training run seed")
    train_parser.add_argument("--dry-run", action="store_true", help="Validate training config and data without API calls")
    train_parser.add_argument("--run-steps", type=int, default=None,
                              help="Run at most N additional steps this launch, then checkpoint and exit; retain the full YAML budget")
    train_parser.add_argument("--resume-run", type=str, default=None,
                              help="Resume the named local run with its optimizer state (directory name only)")

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Run negotiation evaluation")
    eval_parser.add_argument(
        "--config", type=str, default="configs/eval.yaml", help="YAML or TOML config file"
    )
    eval_parser.add_argument(
        "--score-only",
        type=str,
        default=None,
        metavar="LOG_DIR",
        help="Re-score existing logs in LOG_DIR instead of running episodes",
    )
    eval_parser.add_argument(
        "--retry",
        type=str,
        default=None,
        metavar="LOG_DIR",
        help="Retry failed episodes in LOG_DIR and merge results back",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        from src.train import main
        from src.wandb_tracking import reuse_wandb_run

        with reuse_wandb_run():
            main(args.config, seed=args.seed, dry_run=args.dry_run,
                 run_steps=args.run_steps, resume_run=args.resume_run)

    elif args.command == "eval":
        from src.eval.harness import main

        main(args.config, args.score_only, args.retry)


if __name__ == "__main__":
    cli()
