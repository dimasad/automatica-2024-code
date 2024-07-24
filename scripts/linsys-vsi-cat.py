#!/usr/bin/env python3

import os
import pathlib


if __name__ == '__main__':
    # Initialize parameter sets
    mul = set()
    est = set()
    N = set()

    # Iterate over all ".txt" files in the current directory
    for f in pathlib.Path('.').glob('*.txt'):
        parts = f.stem.split('-')
        mul.add(parts[0])
        N.add(int(parts[1]))
        est.add(parts[2])

    # Concatenate the results
    for m in mul:
        for e in est:
            in_files = " ".join([f'{m}-{n}-{e}.txt' for n in sorted(N)])
            cmd =  f'cat {in_files} > linsys-{m}-{e}.plot'
            os.system(cmd)
