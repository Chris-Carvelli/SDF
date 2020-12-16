import sys
from math import sqrt
import argparse
import numpy as np
import PIL
from PIL import Image

import matplotlib.pyplot as plt


def sdf(infile, outfile=None, res=64):
    if outfile is None:
        outfile = f'{infile}_{64}'

    _in = Image.open(infile)

    source = np.array(_in)
    source[source > 0] = 1  # TMP ignores anti-aliasing on source image!
    target = np.empty((res, res, 3), dtype=float)

    # TODO approx properly
    scale_factor = int(4096 / res)
    tile_size = int(scale_factor)
    offset = int(tile_size / 2)
    max_dst = sqrt(pow(tile_size, 2) * 2)

    for i in range(res):
        for j in range(res):
            x = int(i * scale_factor + offset)
            y = int(j * scale_factor + offset)
            val = source[x, y, 0]
            tile = source[x - offset:x + offset, y - offset:y + offset, 0]
            dst_sqrt = dst(tile, val, offset, tile_size, max_dst)
            new_val = norm(dst_sqrt, max_dst)
            target[i, j] = new_val

    plt.imshow(target)
    _out = Image.fromarray((target * 255).astype(np.uint8))
    _out.save(outfile)


def dst(tile: np.ndarray, val, offset, tile_size, max_dst):
    ret = max_dst

    for i in range(tile_size):
        for j in range(tile_size):
            if tile[i, j] != val:
                ret = min(ret, sqrt(pow(offset - i, 2) + pow(offset - j, 2)))

    if val == 0:
        ret *= -1
    return ret


def norm(val, max_val):
    return (val + max_val) / (max_val * 2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a binary, high-res image into a low-res SDF')
    parser.add_argument('infile', metavar='I', type=str)
    parser.add_argument('outfile', nargs='?', metavar='O', type=str)
    parser.add_argument('--res', metavar='R', type=int, default=64)
    args = parser.parse_args()

    sdf(**args.__dict__)
