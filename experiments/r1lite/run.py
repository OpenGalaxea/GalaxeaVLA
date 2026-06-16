import toml
import argparse
import logging

from scheduler.scheduler import Scheduler


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Run scheduler with config")
    parser.add_argument("--config", "-c", type=str, default="config.toml",
                        help="Path to config file (default: config.toml)")
    parser.add_argument("--record", action="store_true",
                        help="Record episodes as video (saved to ./recordings/)")
    parser.add_argument("--binarize-gripper", action="store_true",
                        help="Binarize gripper action at test time (>=threshold → 1.0, else 0.0)")
    parser.add_argument("--gripper-threshold", type=float, default=0.5,
                        help="Threshold for --binarize-gripper (default: 0.5)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show camera images with COT bbox overlay (needs DISPLAY)")
    args = parser.parse_args()

    config = toml.load(args.config)

    recorder = None
    if args.record:
        from utils.episode_recorder import EpisodeRecorder
        recorder = EpisodeRecorder("./recordings")
        logging.info("Recording enabled → ./recordings/")

    scheduler = Scheduler(
        config,
        recorder=recorder,
        binarize_gripper=args.binarize_gripper,
        gripper_threshold=args.gripper_threshold,
        visualize=args.visualize,
    )
    scheduler.run()


if __name__ == "__main__":
    main()
