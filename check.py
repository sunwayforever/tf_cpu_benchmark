#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

stat = defaultdict(lambda: defaultdict(list))
if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <log_file>")
    sys.exit(1)

key = None
x = []
with open(sys.argv[1], "r") as f:
    for line in f.readlines():
        name, p1, p2, n_thread_1, n_thread_2, cost = line.strip().split(":")
        if key is None:
            key = f"{name}:{p1}:{p2}"
        if key == f"{name}:{p1}:{p2}":
            x.append(int(n_thread_1) * int(n_thread_2))

        stat[name][f"{p1}:{p2}"].append(int(cost))

for v in stat.values():
    for k in v.keys():
        first = v[k][0]
        v[k] = [x / first for x in v[k]]

# x = list(range(1, 50, 2))
fig, axs = plt.subplots(len(stat))
if len(stat) == 1:
    axs = [axs]

for i, (k, v) in enumerate(stat.items()):
    m = 1
    for k2, v2 in v.items():
        m = min(v2)
        axs[i].plot(x[:len(v2)], v2, label = f"{k}:{k2}")

    axs[i].legend()
    axs[i].set_xticks(x)
    # axs[i].title.set_text(k)
    axs[i].set_xlabel("#core")
    axs[i].text(0.5, 0.5, f"best: {m:.3f}", horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)
    axs[i].set_ylabel("wall time")

fig.set_size_inches(18.5, 10.5)
fig.savefig(f"{sys.argv[1][:-4]}.png")
