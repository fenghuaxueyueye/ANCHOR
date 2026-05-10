import argparse
import subprocess


def build_base(args):
    return [
        "python",
        "run_gaussian_shading.py",
        "--num", str(args.num),
        "--fpr", str(args.fpr),
        "--model_path", args.model_path,
        "--dataset_path", args.dataset_path,
        "--output_path", args.output_path,
    ]


def maybe_run(cmd, dry_run):
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="ANCHOR sync ablation runner.")
    parser.add_argument("--num", default=100, type=int)
    parser.add_argument("--fpr", default=1e-6, type=float)
    parser.add_argument("--model_path", default="/root/autodl-tmp/sd-2-1-base")
    parser.add_argument("--dataset_path", default="/root/autodl-tmp/sd-prompts")
    parser.add_argument("--output_path", default="./output/ablation")
    parser.add_argument("--attack", default="rst_scale", choices=["none", "rst_scale", "rotation", "crop_resize"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    attack_args = []
    if args.attack == "rst_scale":
        attack_args = ["--rst_scale", "1.25"]
    elif args.attack == "rotation":
        attack_args = ["--rst_rotation_degree", "15"]
    elif args.attack == "crop_resize":
        attack_args = ["--rst_resized_crop_ratio", "0.75"]

    procrustes_cmd = build_base(args) + attack_args
    legacy_cmd = build_base(args) + attack_args + ["--disable_procrustes_sync"]

    maybe_run(procrustes_cmd, args.dry_run)
    maybe_run(legacy_cmd, args.dry_run)


if __name__ == "__main__":
    main()
