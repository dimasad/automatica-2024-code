#!/usr/bin/env python3

import os
import pathlib


if __name__ == '__main__':
    cwd = pathlib.Path.cwd()

    mul = set(f.stem.split('-')[0] for f in cwd.glob('*-nullsys0.txt'))
    for m in mul:
        os.system(f'cat {m}-*-nullsys0.txt > linsys-{m}-nullsys0.plot')
    
    mul = set(f.stem.split('-')[0] for f in cwd.glob('*-initialized.txt'))
    for m in mul:
        os.system(f'cat {m}-*-initialized.txt > linsys-{m}-initialized.plot')
