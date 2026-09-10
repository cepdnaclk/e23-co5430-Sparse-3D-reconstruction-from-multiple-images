"""Export a gsplat checkpoint (.pt) to a standard 3DGS binary PLY.

The PLY carries positions, colors (as SH DC term converted to RGB), opacities
and scales, so the splat can be previewed in MeshLab / CloudCompare / any
viewer that understands the common 3DGS PLY layout.

Usage:
    python gaussian_splatting/export_ply.py results/temple/ckpts/ckpt_29999_rank0.pt \
        temple_splats.ply
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def splats_to_arrays(ckpt: dict) -> dict[str, np.ndarray]:
    splats = ckpt["splats"] if "splats" in ckpt else ckpt
    means = splats["means"].detach().cpu().numpy().astype(np.float32)
    if "rgbs" in splats:
        rgbs = splats["rgbs"].detach().cpu().clamp(0, 1).numpy()
    elif "sh0" in splats:
        # sh0: (N, 1, 3) SH DC coefficient -> RGB = C0 * sh0 + 0.5
        sh0 = splats["sh0"].detach().cpu()
        if sh0.ndim == 3:
            sh0 = sh0[:, 0, :]
        rgbs = (0.28209479177387814 * sh0 + 0.5).clamp(0, 1).numpy()
    else:
        raise SystemExit(f"No color field in ckpt (keys: {list(splats.keys())})")
    opacities = torch.sigmoid(splats["opacities"]).detach().cpu().numpy()
    scales = torch.exp(splats["scales"]).detach().cpu().numpy()
    return {
        "means": means,
        "rgbs": rgbs.astype(np.float32),
        "opacities": opacities.astype(np.float32),
        "scales": scales.astype(np.float32),
    }


def write_ply(path: Path, arrays: dict[str, np.ndarray]) -> None:
    n = len(arrays["means"])
    props = (
        ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "opacity"]
        + [f"scale_{i}" for i in range(3)]
    )
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        for i in range(n):
            f.write(
                struct.pack(
                    "<13f",
                    *arrays["means"][i],
                    0.0,
                    0.0,
                    0.0,
                    *(arrays["rgbs"][i] * 255.0),
                    arrays["opacities"][i],
                    *arrays["scales"][i],
                )
            )
    print(f"[export_ply] wrote {path} ({n} splats)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", type=Path, help="gsplat ckpt_*.pt file")
    ap.add_argument("out", type=Path, help="output .ply path")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    write_ply(args.out, splats_to_arrays(ckpt))


if __name__ == "__main__":
    main()
