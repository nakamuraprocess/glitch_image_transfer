"""
Glitch image transfer.

Simulates a video codec artifact on a still image: a block row or column that
is never refreshed by a keyframe, so it keeps getting copied along one axis and
appears to melt or drip. Both vertical and horizontal smears are applied.

Runs any number of times with a different random seed each time, writing every
result as a lossless PNG into a folder named after the source file. Filenames
are timestamped so repeated runs never overwrite earlier output.

Usage:
    pip install pillow numpy
    python main.py input.png -o ./out -n 10 -b 64 -s 20 --v-dir both

    -> ./out/input/20260721-104500_001.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_BLOCK = 64      # macroblock size, mimics codec block grid
DEFAULT_STREAKS = 14    # how many streaks each smear pass draws


def pick_step(rng, direction, positive):
    """Return +1 or -1 for the requested direction.

    positive names the direction that grows along the axis ("down" or "right").
    "both" leaves the choice to the random generator, per streak.
    """
    if direction == "both":
        return 1 if rng.integers(0, 2) else -1
    return 1 if direction == positive else -1


def vertical_smear(arr, rng, block, streaks=14, max_len=260, direction="down"):
    """Repeat a single row of blocks along the vertical axis: the 'melting' streak.

    Start positions and widths are snapped to the block grid so every streak
    lines up on the same lattice, the way real codec macroblocks do. The grid
    covers the whole canvas, and blocks that run past an edge are clipped
    instead of dropped, so streaks reach the very border with no margin.

    direction is "down", "up", or "both"; with "both" each streak picks a side
    at random.
    """
    h, w = arr.shape[:2]
    cols = -(-w // block)   # ceil, so the partial column at the right edge counts
    rows = -(-h // block)
    out = arr.copy()

    for _ in range(streaks):
        x = int(rng.integers(0, cols)) * block
        y = int(rng.integers(0, rows)) * block
        width = int(rng.integers(1, 6)) * block
        length = int(rng.integers(2, max(3, max_len // block))) * block
        step = pick_step(rng, direction, "down")

        x_end = min(x + width, w)
        src = out[y:min(y + block, h), x:x_end]
        src_h = src.shape[0]
        if src_h == 0 or src.shape[1] == 0:
            continue

        for k in range(0, length, block):
            ty = y + step * (block + k)
            if ty >= h or ty + src_h <= 0:
                break
            # Clip against the edge and take the matching part of the source
            top = max(ty, 0)
            bottom = min(ty + src_h, h)
            out[top:bottom, x:x_end] = src[top - ty:bottom - ty]
    return out


def horizontal_smear(arr, rng, block, streaks=14, max_len=260, direction="right"):
    """Repeat a single column of blocks along the horizontal axis.

    Mirrors vertical_smear: the grid covers the whole canvas and blocks are
    clipped at the edges rather than dropped.

    direction is "right", "left", or "both"; with "both" each streak picks a
    side at random.
    """
    h, w = arr.shape[:2]
    cols = -(-w // block)
    rows = -(-h // block)
    out = arr.copy()

    for _ in range(streaks):
        x = int(rng.integers(0, cols)) * block
        y = int(rng.integers(0, rows)) * block
        height = int(rng.integers(1, 6)) * block
        length = int(rng.integers(2, max(3, max_len // block))) * block
        step = pick_step(rng, direction, "right")

        y_end = min(y + height, h)
        src = out[y:y_end, x:min(x + block, w)]
        src_w = src.shape[1]
        if src_w == 0 or src.shape[0] == 0:
            continue

        for k in range(0, length, block):
            tx = x + step * (block + k)
            if tx >= w or tx + src_w <= 0:
                break
            left = max(tx, 0)
            right = min(tx + src_w, w)
            out[y:y_end, left:right] = src[:, left - tx:right - tx]
    return out


def glitch_once(arr, seed, block, streaks, v_dir, h_dir):
    """Apply one full smear pass to a copy of the source array.

    The streak count is used for both directions, so the total number of
    drawing operations is twice this value.
    """
    rng = np.random.default_rng(seed)
    out = vertical_smear(arr, rng, block, streaks, direction=v_dir)
    out = horizontal_smear(out, rng, block, streaks, direction=h_dir)
    return out


def unique_path(out_dir, index, timestamp):
    """Build a collision-free output path: <YYYYmmdd-HHMMSS>_<nnn>.png

    The timestamp is taken once per run so a batch shares it, and the index
    keeps files within the batch distinct. A numeric suffix is appended in the
    unlikely event the name is already taken.
    """
    base = f"{timestamp}_{index:03d}"
    path = out_dir / f"{base}.png"
    dedup = 1
    while path.exists():
        path = out_dir / f"{base}-{dedup}.png"
        dedup += 1
    return path


def run(path_in, base_dir, count, block, streaks, v_dir, h_dir, seed=None):
    src = Path(path_in)

    # Results are grouped in a folder named after the source file, so batches
    # from different sources never mix.
    out_dir = Path(base_dir) / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(Image.open(src).convert("RGB")).copy()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # One seed sequence per run keeps every variation different but the whole
    # batch reproducible when --seed is given.
    seeds = np.random.default_rng(seed).integers(0, 2**32, size=count)

    for i, s in enumerate(seeds, start=1):
        result = glitch_once(arr, int(s), block, streaks, v_dir, h_dir)
        path_out = unique_path(out_dir, i, timestamp)
        # PNG is lossless, so the hard block edges stay perfectly sharp
        Image.fromarray(result, mode="RGB").save(
            path_out, format="PNG", optimize=True
        )
        print(f"[{i}/{count}] saved: {path_out}")


def main():
    parser = argparse.ArgumentParser(description="Glitch image transfer")
    parser.add_argument("input", help="source image path")
    parser.add_argument(
        "-o", "--output-dir", default="./output",
        help="parent directory; a subfolder named after the input file is created inside it",
    )
    parser.add_argument(
        "-n", "--count", type=int, default=1, help="how many variations to generate"
    )
    parser.add_argument(
        "-b", "--block", type=int, default=DEFAULT_BLOCK,
        help=f"macroblock size in pixels (default {DEFAULT_BLOCK})",
    )
    parser.add_argument(
        "-s", "--streaks", type=int, default=DEFAULT_STREAKS,
        help=f"streaks drawn per direction (default {DEFAULT_STREAKS})",
    )
    parser.add_argument(
        "--v-dir", choices=("down", "up", "both"), default="down",
        help="direction vertical streaks grow (default down)",
    )
    parser.add_argument(
        "--h-dir", choices=("right", "left", "both"), default="right",
        help="direction horizontal streaks grow (default right)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="base seed for reproducible batches"
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be 1 or greater")
    if args.block < 1:
        parser.error("--block must be 1 or greater")
    if args.streaks < 0:
        parser.error("--streaks must be 0 or greater")

    run(args.input, args.output_dir, args.count, args.block,
        args.streaks, args.v_dir, args.h_dir, args.seed)


if __name__ == "__main__":
    sys.exit(main())