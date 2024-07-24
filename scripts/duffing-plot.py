#!/usr/bin/env python3


import argparse
import pathlib

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=argparse.FileType('r'))
    parser.add_argument('--iplot', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    entry_labels = "tf N opt_time a b d g logstd_w logstd_v x0 x1".split()

    data = {}
    with args.datafile as f:
        for line in f:
            tokens = line.split()
            prob = tokens[0]
            case = "sk-"+tokens[1] if prob == "smoother-kernel" else prob
            tf = float(tokens[2])
            entry = np.array(list(map(float, tokens[2:])))
            data.setdefault(case, {}).setdefault(tf, []).append(entry)
    num_entries = len(entry)

    low = {}
    high = {}
    mean = {}
    med = {}

    for case, entries in data.items():
        low[case] = np.array([np.quantile(a, 0.25, 0) for a in entries.values()])
        high[case] = np.array([np.quantile(a, 0.75, 0) for a in entries.values()])
        med[case] = np.array([np.median(a, 0) for a in entries.values()])
        mean[case] = np.array([np.mean(a, 0) for a in entries.values()])

    if args.iplot:
        import cycler
        from matplotlib import pyplot as plt
        cycle = (
            cycler.cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k']) +
            cycler.cycler(marker=['o', 's', 'D', '^', 'v', 'p', 'P'])
        )
            
        with plt.ion():
            for i in range(2, num_entries):
                plt.figure(i)
                ax = plt.gca()
                ax.cla()
                ax.set_prop_cycle(cycle)
                for case in data:
                    yerr = np.array([low[case][:,i], high[case][:,i]])
                    eps = 1+(0.5 + np.random.rand(*yerr[0].shape))/4
                    ax.errorbar(
                        med[case][:,0]*eps, med[case][:,i], yerr=yerr,
                        label=case, ls='', 
                    )

                plt.title(entry_labels[i])
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()

    cat_labels = [
        *[f"{l}_med" for l in entry_labels],
        *[f"{l}_low" for l in entry_labels],
        *[f"{l}_high" for l in entry_labels],
        *[f"{l}_mean" for l in entry_labels],
    ]
    for case in data:
        a = np.c_[med[case], low[case], high[case], mean[case]]
        np.savetxt(f"{case}.plot", a, header=" ".join(cat_labels), comments='')
