import argparse
import math

import torch

from watermark import weighted_procrustes_2d


def make_grid_points(grid_size=8, spacing=4.0):
    coords = []
    offset = (grid_size - 1) * spacing / 2.0
    for row in range(grid_size):
        for col in range(grid_size):
            coords.append([col * spacing - offset, row * spacing - offset])
    return torch.tensor(coords, dtype=torch.float32)


def apply_similarity(points, scale, angle_deg, tx, ty):
    theta = math.radians(angle_deg)
    rot = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([tx, ty], dtype=torch.float32)
    return scale * points.matmul(rot.t()) + translation


def run_once(args):
    torch.manual_seed(args.seed)
    src = make_grid_points(args.grid_size, args.spacing)
    dst_clean = apply_similarity(src, args.scale, args.angle, args.tx, args.ty)
    dst = dst_clean + torch.randn_like(dst_clean) * args.noise_std

    if args.crop_keep < 1.0:
        keep_n = max(3, int(round(src.shape[0] * args.crop_keep)))
        perm = torch.randperm(src.shape[0])[:keep_n]
        src = src[perm]
        dst = dst[perm]

    estimate = weighted_procrustes_2d(src, dst)
    scale_err = abs(estimate["scale"] - args.scale)
    angle_err = abs(((estimate["angle"] - args.angle + 180.0) % 360.0) - 180.0)
    tx_err = abs(float(estimate["translation"][0].item()) - args.tx)
    ty_err = abs(float(estimate["translation"][1].item()) - args.ty)

    print("=== Procrustes sync synthetic check ===")
    print(f"points: {src.shape[0]}")
    print(f"target scale={args.scale:.6f}, estimated={estimate['scale']:.6f}, error={scale_err:.6e}")
    print(f"target angle={args.angle:.6f}, estimated={estimate['angle']:.6f}, error={angle_err:.6e}")
    print(f"target tx={args.tx:.6f}, estimated={float(estimate['translation'][0].item()):.6f}, error={tx_err:.6e}")
    print(f"target ty={args.ty:.6f}, estimated={float(estimate['translation'][1].item()):.6f}, error={ty_err:.6e}")
    print(f"weighted rmse={estimate['rmse']:.6e}")

    if args.noise_std == 0.0:
        assert scale_err < 1e-5
        assert angle_err < 1e-4
        assert tx_err < 1e-4
        assert ty_err < 1e-4


def main():
    parser = argparse.ArgumentParser(description="Synthetic Procrustes synchronization test for ANCHOR.")
    parser.add_argument("--grid_size", default=8, type=int)
    parser.add_argument("--spacing", default=4.0, type=float)
    parser.add_argument("--scale", default=1.25, type=float)
    parser.add_argument("--angle", default=17.0, type=float)
    parser.add_argument("--tx", default=3.0, type=float)
    parser.add_argument("--ty", default=-2.0, type=float)
    parser.add_argument("--noise_std", default=0.0, type=float)
    parser.add_argument("--crop_keep", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
