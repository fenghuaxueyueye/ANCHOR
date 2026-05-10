import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run vanilla null calibration for ANCHOR thresholds.")
    parser.add_argument("--num", default=1000, type=int)
    parser.add_argument("--fpr", default=1e-6, type=float)
    parser.add_argument("--model_path", default="/root/autodl-tmp/sd-2-1-base")
    parser.add_argument("--dataset_path", default="/root/autodl-tmp/sd-prompts")
    parser.add_argument("--output_path", default="./output/null_calibration")
    parser.add_argument("--disable_procrustes_sync", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cmd = [
        "python",
        "run_gaussian_shading.py",
        "--eval_mode", "vanilla",
        "--num", str(args.num),
        "--fpr", str(args.fpr),
        "--model_path", args.model_path,
        "--dataset_path", args.dataset_path,
        "--output_path", args.output_path,
    ]
    if args.disable_procrustes_sync:
        cmd.append("--disable_procrustes_sync")

    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
