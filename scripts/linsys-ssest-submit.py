#!/usr/bin/env python3

import os
import pathlib


if __name__ == '__main__':
    # Set GLOBAL environment variables
    os.environ['SBATCH_GPUS'] = '0'
    os.environ['SBATCH_PARTITION'] = 'comm_small_day'
    os.environ['ARG_N'] = '10000'

    # Iterate over output MAT files in the current directory
    for f in pathlib.Path('.').glob('*-gvi.mat'):
        baseid = f.name[:-8]

        # Set the MAT file to be used
        os.environ['ARG_MATFILE'] = str(f)

        # Run the job with default initialization
        id = f'{baseid}-ssest-initialized'
        os.environ['SBATCH_JOB_NAME'] = f'{id}'
        os.environ['ARG_OUTFILE'] = f'{id}.txt'
        os.environ['ARG_NULLSYS0'] = 'false'
        os.system('sbatch linsys-ssest-run')

        # Run the job with null initial system
        id = f'{baseid}-ssest-nullsys0'
        os.environ['SBATCH_JOB_NAME'] = f'{id}'
        os.environ['ARG_OUTFILE'] = f'{id}.txt'
        os.environ['ARG_NULLSYS0'] = 'true'
        os.system('sbatch linsys-ssest-run')
