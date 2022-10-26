#!/usr/bin/env python3
import matplotlib.pyplot as plt

from collections import defaultdict

stat = defaultdict(lambda: defaultdict(list))

with open("eigen.log", "r") as f:
    for line in f.readlines():
        name, p1, p2, _, n_thread, cost = line.strip().split(":")
        stat[name][f"{p1}:{p2}"].append(int(cost))

for v in stat.values():
    for k in v.keys():
        first = v[k][0]
        v[k] = [x / first for x in v[k]]

x = list(range(1, 50, 2))
fig, axs = plt.subplots(len(stat))

for i, (k, v) in enumerate(stat.items()):
    for k2, v2 in v.items():
        axs[i].plot(x, v2, label = k2)

    axs[i].legend()
    axs[i].set_xticks(x)
    axs[i].title.set_text(k)

fig.set_size_inches(18.5, 10.5)
fig.savefig("output.png")
